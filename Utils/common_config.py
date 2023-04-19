import pathlib
import yaml
import time
ROOT = pathlib.Path(__file__).parent.parent

class Config:
    def __init__(self , filename = None) -> None:
        self.readconfig('defaultruntime.yaml')
        self.readconfig(filename)

    def readconfig(self , filename) -> None:
        filepath = str(ROOT / 'RunTimeConf' / filename)
        self.logger_file = str(ROOT / 'RunLogger' / (filename+time.strftime("%d_%m_%Y_%H_%M_%S")))
        f = open(filepath , 'r', encoding='utf-8')
        desc = yaml.load(f.read(),Loader=yaml.FullLoader)
        f.close()
        for key , value in desc.items():
            setattr(self,key,value)

        self.datapath = str(ROOT / 'DataSource' / desc['dataset'])
        self.cachepath = str(ROOT / 'Cache' /  (desc['dataset'] + '_' + str(self.batch_size) + '_' + '_'.join(self.split)) )
        self.savedpath = str(ROOT / 'Saved' / (desc['model'] + desc['dataset'] ))
        self.logdir = str(ROOT / 'Log')
        with open(str(ROOT / 'MetaData' / (desc['dataset'] + '.yaml') ) , 'r') as f:
            descb = yaml.load(f.read(),Loader=yaml.FullLoader)
            self.feature_stastic = descb['feature_stastic']
            self.feature_default = descb['feature_defaults']
    