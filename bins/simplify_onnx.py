# -*- coding: utf-8 -*-
"""
@Time : 2021/9/24 0:32 
@Author : Zhi QIN 
@File : simplify_onnx.py 
@Software: PyCharm
@Brief : 
"""

import onnx
from onnxsim import simplify

onnx_model = onnx.load(f=r'D:\PythonCode\BiSeNet\results\ONNX\model.onnx')

model_simp, check = simplify(onnx_model)

assert check, "Simplified ONNX model could not be validated"

onnx.save(model_simp, r'D:\PythonCode\BiSeNet\results\ONNX\model_simp.onnx')

print('finished exporting onnx')
