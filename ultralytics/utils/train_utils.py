import time
import onnx
import inspect
import logging
import numpy as np
import onnxruntime as ort

import torch

import torchvision
from torchvision import datasets
from torchvision.models.resnet import resnet18, resnet50
import torchvision.transforms as transforms

logging.basicConfig(level=logging.INFO)


def cifar10_data_loaders(data_path, train_batch_size = 32, eval_batch_size = 32):
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.Resize(224),         # ResNet-18 输入尺寸为 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 下载并加载数据集
    dataset = datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform
    )
    dataset_test = datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform
    )

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader, data_loader_test


def imagenet_data_loaders(data_path, train_batch_size = 32):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = torchvision.datasets.ImageNet(
        data_path, split="train", transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    dataset_test = torchvision.datasets.ImageNet(
        data_path, split="val", transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler)

    return data_loader, data_loader_test


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_np(output: np.ndarray, target: np.ndarray):
    max_indices = np.argsort(output, axis=1)[:, ::-1]
    top5 = 100 * np.equal(max_indices[:, :5], target[:, np.newaxis]).sum(axis=1).mean()
    top1 = 100 * np.equal(max_indices[:, 0], target).mean()
    return top1, top5


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified
    values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate_np(sess, data_loader_test, total_size=None):
    _logger = logging.getLogger("resnet:")
    # _logger.setLevel(logging.INFO)

    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    for i, (image, target) in enumerate(data_loader_test):
        if total_size is not None and i >= total_size:
            return top1, top5

        image = image.numpy()
        target = target.numpy()
        if isinstance(sess, ort.InferenceSession):
            output = sess.run(None, {"x_0": image})[0]
        elif inspect.isfunction(sess):
            output = sess(image)
        else:
            assert False
        batch = output.shape[0]

        acc1, acc5 = accuracy_np(output, target)

        top1.update(acc1.item(), batch)
        top5.update(acc5.item(), batch)

        batch_time.update(time.time() - end)
        end = time.time()
        _logger.info(
            "Test: [{0:>4d}/{1}]  "
            "Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
            "Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  "
            "Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})".format(
                i,
                len(data_loader_test),
                batch_time=batch_time,
                rate_avg=batch / batch_time.avg,
                top1=top1,
                top5=top5,
            )
        )
    return top1, top5


def evaluate(model, data_loader_test, total_size=None):
    _logger = logging.getLogger("resnet:")
    if isinstance(model, torch.fx.graph_module.GraphModule):
        torch.ao.quantization.move_exported_model_to_eval(model)

    device = torch.device("cuda")
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader_test):
            if total_size is not None and i >= total_size:
                return top1, top5
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            batch = output.shape[0]

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], batch)
            top5.update(acc5[0], batch)

            batch_time.update(time.time() - end)
            end = time.time()
            _logger.info(
                "Test: [{0:>4d}/{1}]  "
                "Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                "Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  "
                "Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})".format(
                    i,
                    len(data_loader_test),
                    batch_time=batch_time,
                    rate_avg=batch / batch_time.avg,
                    top1=top1,
                    top5=top5,
                )
            )

    return top1, top5


def load_model(model_file, name = "resnet18"):
    if name == "resnet18":
        model = resnet18(weights=None)
    elif name == "resnet50":
        model = resnet50(weights=None)
    else:
        assert False
    state_dict = torch.load(model_file, weights_only=True)
    model.load_state_dict(state_dict)
    return model


def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    # Note: do not call model.train() here, since this doesn't work on an exported model.
    # Instead, call `torch.ao.quantization.move_exported_model_to_train(model)`, which will
    # be added in the near future
    top1 = AverageMeter()
    top5 = AverageMeter()
    avgloss = AverageMeter()

    cnt = 0
    for i, (image, target) in enumerate(data_loader):
        start_time = time.time()
        print('.', end = '')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))

        print(f"Training: [{i}/{len(data_loader)}] Loss: {avgloss.avg:.3f} Acc@1: {top1.avg:.3f} Acc@5: {top5.avg:.3f}")
        if cnt >= ntrain_batches:
        #     print('Loss', avgloss.avg)

        #     print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #           .format(top1=top1, top5=top5))
            return

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=top1, top5=top5))
    return


def dynamo_export(model, inputs, onnx_path):
    onnx_program = torch.onnx.export(model, inputs, output_names=['output'], dynamo=True, opset_version=21)
    onnx_program.optimize()
    onnx_program.save(onnx_path)
    print(f"save onnx model to [{onnx_path}] Successfully!")


def onnx_simplify(onnx_path, sim_path):
    from onnxslim import slim

    model = onnx.load(onnx_path)
    model_simp = slim(model)
    onnx.save(model_simp, sim_path)
    print(f"save onnx model to [{sim_path}] Successfully!")
