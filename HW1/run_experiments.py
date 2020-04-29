from mnist_scratch import main

batchsize = [1,10,100,1000]
frac_dset = [50000/60000., 10000/60000., 1000/60000., 500/60000., 100/60000.]

for i in range(len(batchsize)):
    print('Running exp w/ batch size %d' % batchsize[i])
    main(batchsize=batchsize[i], lr=0.001, epochs=10, frac_dset=5/6., fname='exp_bs_%d.txt'%batchsize[i])

for i in range(len(frac_dset)):
    print('Running exp w/  %d of train set' % int(frac_dset[i]*60000))
    main(batchsize=100, lr=0.001, epochs=10, frac_dset=frac_dset[i], fname='exp_df_%d.txt'%int(frac_dset[i]*60000))