import fire

from model.s2_network import S2Network
from model.s3_network import S3Network

if __name__ == '__main__':
    fire.Fire({
        's2_network': S2Network.train,
        's3_network': S3Network.train
    })
