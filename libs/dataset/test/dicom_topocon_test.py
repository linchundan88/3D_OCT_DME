
import pydicom

filename = '/disk1/3D_OCT_DME/original/Topocon/Topocon/累及中央的黄斑水肿/02-000003_20160601_120201_OPT_R_001.dcm'
ds2 = pydicom.dcmread(filename)
array1 = ds2.pixel_array  #(128, 885, 512)  (D,H,W)

print(array1.shape)