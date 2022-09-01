1.运行train.py，会在当前目录下生成:
	best_resNet34.pth,训练好的模型;
	class_indices.json,带类别及索引的json文件
	training_resilt.png,训练结果图片。
2.运行images_mining.py会在当前目录下生成{class}_pictures_you_want文件夹，class为目标类别，在代码中修改。
3.运行SITF_Find_Same_Object.py会在当前目录下生成related_pictures文件夹，里边放有找到的频繁性图片。