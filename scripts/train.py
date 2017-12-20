import fire

# from model.s2_network import S2Network
# # from model.s3_network import S3Network
# #
# # # import model.convdeep_4l_network
# # # import model.deep_4l_network
# # import model
from model import s2_network, s3_network, deep_4l_network, convdeep_4l_network

if __name__ == '__main__':
    fire.Fire({
        's2_network': s2_network.S2Network.train,
        's3_network': s3_network.S3Network.train,
        'deep_4l_network': deep_4l_network.Network.train,
        'convdeep_4l_network': convdeep_4l_network.Network.train
    })
