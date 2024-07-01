from . import model_utils

def buildModel(args):
    print('Creating Model %s' % (args.model))
    if args.model == 'ESSENCE_Net':
        from models.ESSENCE_Net import ESSENCENET
        model = ESSENCENET()
    else:
        raise Exception("=> Unknown Model '{}'".format(args.model))
    
    if args.cuda: 
        model = model.cuda()

    if args.retrain: 
        print("=> using pre-trained model %s" % (args.retrain))
        model_utils.loadCheckpoint(args.retrain, model, cuda=args.cuda)

    print(model)
    print("=> Model Parameters: %d" % (model_utils.get_n_params(model)))
    return model
