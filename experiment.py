from train import main
import shutil
import os
from loguru import logger

TEST_TIME = 3

TEST_LIST = [
            #  0, 
            #  1, 
            #  2, 
             3,
]

"""毛体积估计-普通钣金"""
def test_material():
    if os.path.exists(r"E:\Project\AutoPricing\datasets\atwmaterial\scaler.joblib"):
        os.remove(r"E:\Project\AutoPricing\datasets\atwmaterial\scaler.joblib")
    shutil.copy2(r"E:\Project\AutoPricing\datasets\atwmaterial\v5\train.txt", 
                r"E:\Project\AutoPricing\datasets\atwmaterial\train.txt")
    shutil.copy2(r"E:\Project\AutoPricing\datasets\atwmaterial\v5\test.txt", 
                r"E:\Project\AutoPricing\datasets\atwmaterial\test.txt")
    shutil.copy2(r"E:\Project\AutoPricing\datasets\atwmaterial\v5\val.txt", 
                r"E:\Project\AutoPricing\datasets\atwmaterial\val.txt")
    logger.info("毛体积估计-普通钣金 scale noscheduler")
    for i in range(TEST_TIME):
        main(max_epochs=800,
            center_and_scale=True,
            edge_input_dim=3,
            face_input_dim=3,
            vars_dim=7,
            batch_size=256,
            crv_emb_dim=64,
            srf_emb_dim=64,
            graph_emb_dim=128,
            datasetDir=r"E:\Project\AutoPricing\datasets",
            mode="train",
            dataset="atwmaterial",
            lossfn='L1',
            init_lr=0.01,
            scheduler=None,
            scaler_file="scaler.joblib")  
        
    # logger.info("毛体积估计-普通钣金 channel/2 scale noscheduler")
    # for i in range(TEST_TIME):
    #     main(max_epochs=500,
    #         center_and_scale=True,
    #         edge_input_dim=3,
    #         face_input_dim=3,
    #         vars_dim=7,
    #         crv_emb_dim=32,
    #         srf_emb_dim=32,
    #         graph_emb_dim=64,
    #         datasetDir=r"E:\Project\AutoPricing\datasets",
    #         mode="train",
    #         dataset="atwmaterial",
    #         lossfn='L1',
    #         scheduler=None,
    #         scaler_file="scaler.joblib") 



"""加工价格估计-普通钣金"""
def test_price():
    if os.path.exists(r"E:\Project\AutoPricing\datasets\atwcad\scaler.joblib"):
        os.remove(r"E:\Project\AutoPricing\datasets\atwcad\scaler.joblib")
    shutil.copy2(r"E:\Project\AutoPricing\datasets\atwcad\v5\train.txt", 
                r"E:\Project\AutoPricing\datasets\atwcad\train.txt")
    shutil.copy2(r"E:\Project\AutoPricing\datasets\atwcad\v5\test.txt", 
                r"E:\Project\AutoPricing\datasets\atwcad\test.txt")
    shutil.copy2(r"E:\Project\AutoPricing\datasets\atwcad\v5\val.txt", 
                r"E:\Project\AutoPricing\datasets\atwcad\val.txt")
    if os.path.exists(r"E:\Project\AutoPricing\datasets\atwcad\v5\scaler.joblib"):
        shutil.copy2(r"E:\Project\AutoPricing\datasets\atwcad\v5\scaler.joblib", 
                    r"E:\Project\AutoPricing\datasets\atwcad\scaler.joblib")
    
    logger.info("加工价格估计-普通钣金 Validation w. scale (88.04)")
    main(mode="test", 
         dataset="atwcad",
         center_and_scale=True,
         checkpointPath=r"E:\LGJ\program\UV-Net\results\regression\0620\pric-88.04\epoch=469-val_loss=0.0870-val_acc=0.8804.ckpt",
         scaler_file="scaler.joblib")
    logger.info("加工价格估计-普通钣金 Validation w.o scale (86.66)")
    main(mode="test", 
         dataset="atwcad",
         center_and_scale=False,
         checkpointPath=r"E:\LGJ\program\UV-Net\results\regression\0623\141959\epoch=419-val_loss=2.0745-val_acc=0.8666.ckpt",)



    
    # logger.info("加工价格估计-普通钣金 scale noscheduler")
    # for i in range(TEST_TIME):
    #     main(max_epochs=500,
    #         center_and_scale=True,
    #         edge_input_dim=3,
    #         face_input_dim=3,
    #         vars_dim=8,
    #         crv_emb_dim=64,
    #         srf_emb_dim=64,
    #         graph_emb_dim=128,
    #         datasetDir=r"E:\Project\AutoPricing\datasets",
    #         mode="train",
    #         dataset="atwcad",
    #         lossfn='L1',
    #         scheduler=None,
    #         scaler_file="scaler.joblib") 
    
    # logger.info("加工价格估计-普通钣金 scale noscheduler")
    # for i in range(TEST_TIME):
    #     main(max_epochs=500,
    #         center_and_scale=True,
    #         edge_input_dim=3,
    #         face_input_dim=3,
    #         vars_dim=8,
    #         crv_emb_dim=64,
    #         srf_emb_dim=64,
    #         graph_emb_dim=128,
    #         datasetDir=r"E:\Project\AutoPricing\datasets\atwcad",
    #         mode="train",
    #         dataset="atwcad",
    #         lossfn='L1',
    #         scheduler=None,
    #         scaler_file="scaler.joblib") 
    

"""毛体积估计-焊接钣金"""
def test_material_weld():
    if os.path.exists(r"E:\Project\AutoPricing\datasets\atwmaterial\scaler.joblib"):
        os.remove(r"E:\Project\AutoPricing\datasets\atwmaterial\scaler.joblib")
    shutil.copy2(r"E:\Project\AutoPricing\datasets\atwmaterial\v5_1\train.txt", 
                r"E:\Project\AutoPricing\datasets\atwmaterial\train.txt")
    shutil.copy2(r"E:\Project\AutoPricing\datasets\atwmaterial\v5_1\test.txt", 
                r"E:\Project\AutoPricing\datasets\atwmaterial\test.txt")
    shutil.copy2(r"E:\Project\AutoPricing\datasets\atwmaterial\v5_1\val.txt", 
                r"E:\Project\AutoPricing\datasets\atwmaterial\val.txt")
    logger.info("毛体积估计-焊接钣金 scale noscheduler channel*2")
    for i in range(TEST_TIME):
        main(max_epochs=800,
            center_and_scale=True,
            edge_input_dim=3,
            face_input_dim=3,
            vars_dim=7,
            batch_size=64,
            crv_emb_dim=128,
            srf_emb_dim=128,
            graph_emb_dim=256,
            datasetDir=r"E:\Project\AutoPricing\datasets",
            mode="train",
            dataset="atwmaterial",
            lossfn='L1',
            scheduler=None,
            scaler_file="scaler.joblib") 

    
"""加工价格估计-焊接钣金"""
def test_price_weld():
    if os.path.exists(r"E:\Project\AutoPricing\datasets\atwcad\scaler.joblib"):
        os.remove(r"E:\Project\AutoPricing\datasets\atwcad\scaler.joblib")
    shutil.copy2(r"E:\Project\AutoPricing\datasets\atwcad\v5_1\train.txt", 
                r"E:\Project\AutoPricing\datasets\atwcad\train.txt")
    shutil.copy2(r"E:\Project\AutoPricing\datasets\atwcad\v5_1\test.txt", 
                r"E:\Project\AutoPricing\datasets\atwcad\test.txt")
    shutil.copy2(r"E:\Project\AutoPricing\datasets\atwcad\v5_1\val.txt", 
                r"E:\Project\AutoPricing\datasets\atwcad\val.txt")
    if os.path.exists(r"E:\Project\AutoPricing\datasets\atwcad\v5_1\scaler.joblib"):
        shutil.copy2(r"E:\Project\AutoPricing\datasets\atwcad\v5_1\scaler.joblib", 
                    r"E:\Project\AutoPricing\datasets\atwcad\scaler.joblib")
    
    # logger.info("加工价格估计-焊接钣金 Validation w. scale (84.82)")
    # main(mode="test", 
    #      dataset="atwcad",
    #      center_and_scale=True,
    #      batch_size=64,
    #      checkpointPath=r"E:\LGJ\program\UV-Net\results\regression\0701\172311\epoch=534-val_loss=0.1155-val_acc=0.8383.ckpt",
    #      scaler_file="scaler.joblib")
    logger.info("加工价格估计-焊接钣金 Validation w.o scale (82.16)")
    main(mode="test", 
         dataset="atwcad",
         center_and_scale=False,
         checkpointPath=r"E:\LGJ\program\UV-Net\results\regression\0624\013642\epoch=479-val_loss=7.8872-val_acc=0.8216.ckpt",)
    
    logger.info("加工价格估计-焊接钣金 scale noscheduler channel*2")
    for i in range(TEST_TIME):
        main(max_epochs=800,
            center_and_scale=True,
            edge_input_dim=3,
            face_input_dim=3,
            vars_dim=8,
            batch_size=64,
            crv_emb_dim=128,
            srf_emb_dim=128,
            graph_emb_dim=256,
            datasetDir=r"E:\Project\AutoPricing\datasets",
            mode="train",
            dataset="atwcad",
            lossfn='L1',
            scheduler=None,
            scaler_file="scaler.joblib")
        



if __name__ == '__main__':
    if 0 in TEST_LIST:
        test_material()
    if 1 in TEST_LIST:
        test_price()
    if 2 in TEST_LIST:
        test_material_weld()
    if 3 in TEST_LIST:
        test_price_weld()