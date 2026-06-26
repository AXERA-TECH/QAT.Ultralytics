from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# COCOMAP
tmp_gts_path = './tmp_gts.json'
tmp_preds_path = './ax_preds-qat11.json'
print(f'tmp_preds_path {tmp_preds_path}')
cocoGt = COCO(tmp_gts_path)
cocoDt = cocoGt.loadRes(tmp_preds_path)
imgIds = sorted(cocoGt.getImgIds())

# running evaluation
cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
res = {"output": {"mAP@0.5": cocoEval.stats[1] * 100, "mAP@0.5:0.95": cocoEval.stats[0] * 100}}
print(res)