import torch
from models import model_utils
from utils import eval_utils, time_utils 


def fab(a,b,input, model):
    if (input[0].shape[2] // a)%4 != 0:
        h = input[0].shape[2] // a +4- (input[0].shape[2] // a)%4
    else:
        h = input[0].shape[2] // a
    if (input[0].shape[3] // b)%4 != 0:
        w = input[0].shape[3] // b + 4- (input[0].shape[3] //b)%4
    else:
        w = input[0].shape[3] // b

    out_varh =[0]*b
    out_varw = [0]*a
    for i in range(a):
        for j in range(b):
            inputf = [input[0][:, :, max(0,h*i-8):min(h*(i+1) + 8,input[0].shape[2]), max(0,w*j-8):min(w*(j+1) + 8,input[0].shape[3])],
                      input[1][:, :, max(0,h*i-8):min(h*(i+1) + 8,input[1].shape[2]), max(0,w*j-8):min(w*(j+1) + 8,input[1].shape[3])]]
            out_varf,shading_out, shading_out_intra= model(inputf)
            c = [8] * a; c[0]   = 0
            d = [8] * a; d[a-1] = 0
            e = [8] * b; e[0] = 0
            f = [8] * b; f[b - 1] = 0
            out_varh[j]=out_varf[:,:,c[i]:out_varf.shape[2]  - d[i],e[j]:out_varf.shape[3]  - f[j]]
        out_varw[i] = torch.cat(out_varh, 3)
    out_var = torch.cat(out_varw, 2)
    del out_varh, out_varw
    return out_var


def test(args, split, loader, model, log,epoch, recorder):
    model.eval()
    print('---- Start %s: %d batches ----' % (split, len(loader)))
    timer = time_utils.Timer(args.time_sync)
    with torch.no_grad():
        for i, sample in enumerate(loader):
            data = model_utils.parseData(args, sample, timer)
            input = model_utils.getInput(data)
            # out_var = fab(1, 2, input, model)
            # out_var, shading_out, shading_out_intra = model(input)
            timer.updateTime('data_p')
            if i%10== 1 or 2  :
                out_var=fab(1,2,input,model)
            else:
                out_var, shading_out, shading_out_intra = model(input)
            timer.updateTime('Forward')

            acc, error_map = eval_utils.calNormalAcc(data['tar'].data, out_var.data, data['m'].data)
            data['error_map'] = error_map['angular_map']
            recorder.updateIter(split, acc.keys(), acc.values())

            iters = i + 1
            opt = {'split':split, 'epoch': epoch,   'iters':iters, 'batch':len(loader),
                        'timer':timer, 'recorder': recorder}
            log.printItersSummary(opt)

            results= prepareSave(data, out_var)
            log.saveImgResults(results, split, data['obj'][0])

    opt = {'split': split,  'epoch': epoch,  'recorder': recorder}
    log.printEpochSummary(opt)

def prepareSave(data, pred):
    results = [(data['tar'].data + 1) / 2 * data['m'].data.expand_as(pred.data) + 255 * (
                1 - data['m'].data.expand_as(pred.data))]
    pred_n = (pred.data + 1) / 2
    masked_pred = pred_n * data['m'].data.expand_as(pred.data) + 255 * (1 - data['m'].data.expand_as(pred.data))
    res_n = [masked_pred, data['error_map']]
    results += res_n
    return results
