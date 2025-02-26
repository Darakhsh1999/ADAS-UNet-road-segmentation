import torch

def test_metrics():
    N = 100
    A = torch.zeros((N,N))
    B = torch.zeros((N,N))

    A[0:N//2,0:N//2] = 1
    B[0:N//2,0:N//2] = 1

    cm = confusion_matrix(A,B)
    ddict = calculate_metrics(cm)
    print(ddict)

    A = torch.ones((N,N))
    B = torch.zeros((N,N))

    cm = confusion_matrix(A,B)
    ddict = calculate_metrics(cm)
    print(ddict)

    A = torch.randint(0,2,(N,N))
    B = torch.randint(0,2,(N,N))

    cm = confusion_matrix(A,B)
    ddict = calculate_metrics(cm)
    print(ddict)

def IOU(pred:torch.Tensor, gt:torch.Tensor) -> float:
    intersection = torch.bitwise_and(pred,gt)
    union = torch.bitwise_or(pred,gt)
    return (intersection.sum() / union.sum()).item()

def confusion_matrix(pred:torch.Tensor, gt:torch.Tensor) -> torch.Tensor:
    TP = torch.logical_and(pred, gt).sum()
    TN = torch.logical_not(torch.logical_or(pred,gt)).sum()
    FP = torch.logical_and(pred, torch.logical_not(gt)).sum()
    FN = torch.logical_and(torch.logical_not(pred), gt).sum()
    return torch.Tensor([[TP,FN],[FP,TN]])

def calculate_metrics(cm) -> dict:

    metric_dict = {}

    # Calculate metrics
    metric_dict["iou"] = cm[0,0] / (cm[0,0]+cm[0,1]+cm[1,0]) if ((cm[0,0]+cm[0,1]+cm[1,0]) != 0) else 0
    metric_dict["dice"] = 2*cm[0,0] / (2*cm[0,0]+cm[0,1]+cm[1,0]) if (2*(cm[0,0]+cm[0,1]+cm[1,0]) != 0) else 0
    metric_dict["precision"] = cm[0,0] / (cm[0,0]+cm[1,0]) if ((cm[0,0]+cm[1,0]) != 0) else 0
    metric_dict["recall"] = cm[0,0] / (cm[0,0]+cm[0,1]) if ((cm[0,0]+cm[0,1]) != 0) else 0

    return metric_dict


if __name__ == "__main__":
    
    test_metrics()