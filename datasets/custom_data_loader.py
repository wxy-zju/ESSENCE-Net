import torch.utils.data


def benchmarkLoader(args):
    print("=> fetching img pairs in data/%s" % (args.benchmark))
    if args.benchmark == 'DiLiGenT_main':
        from datasets.DiLiGenT_main import DiLiGenT_main
        test_set  = DiLiGenT_main(args, 'test')
    else:
        raise Exception('Unknown benchmark')

    print('\t Found Benchmark Data: %d samples' % (len(test_set)))
    print('\t Test Batch %d' % (args.test_batch))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch,
                        num_workers=args.workers, pin_memory=args.cuda, shuffle=False)
    return test_loader
