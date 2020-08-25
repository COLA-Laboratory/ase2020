from Utils.File import fnameList, create_dir
from Utils.helper import MfindCommonMetric
from Auto_CPDP.estimators import Cpdp
import time, warnings

warnings.filterwarnings('ignore')



repeat = 20


if __name__ == '__main__':
    begin_num = 1
    end_num = 20

    flist = []
    group = sorted([
        'ReLink',
        'AEEEM',
        'JURECZKO'
    ])

    resDir = create_dir('resAuto_CPDP10')
    hisDir = create_dir('hisAuto_CPDP10')

    for i in range(len(group)):
        tmp = []
        fnameList('data/' + group[i], tmp)
        tmp = sorted(tmp)
        flist.append(tmp)

    for c in range(begin_num, end_num + 1):
        if c in range(6):
            tmp = flist[0].copy()
            target = tmp.pop(c - 1)
        if c in range(6, 18):
            tmp = flist[1].copy()
            target = tmp.pop(c - 6)
        if c in range(18, 21):
            tmp = flist[2].copy()
            target = tmp.pop(c - 18)

        Xsource, Lsource, Xtarget, Ltarget, loc = MfindCommonMetric(tmp, target, split=True)
        for k in range(repeat):
            stime = time.time()
            cpdp = Cpdp(
                time_left_for_this_task=3600,
                    per_run_time_limit=360,
                # maxFE=10,
                repeat=10
            )
            cpdp.fit(Xsource, Lsource, Xtarget, Ltarget, loc)
            with open(resDir + target.split('/')[-1][:-5] + '.txt', 'a+') as f:
                print(cpdp.incument_, file=f)
                print('time:', time.time() - stime, file=f)
                print('--------------------', file=f)
            with open(hisDir + target.split('/')[-1][:-5] + '.txt', 'a+') as f:
                print(cpdp.trajectory_, file=f)
                print('--------------------', file=f)

    print('done!')



