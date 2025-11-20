
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# 将数据集文件或其他文件上传到魔搭社区仓库

# 在 SDK 中完成访问令牌登陆
from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = 'ms-51e64baa-8d60-43ae-b4a8-43883956bde8'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

# 使用文件接口上传#
# 假定您已经创建好数据集仓库，账户名是rickxt，数据集英文名称为GPT-2-datasets。
# 上传数据文件夹
'''
参数说明

字段名	必填	类型	描述
repo_id	是	str	数据集ID，确保您的访问令牌具有上传至对应仓库的权限。
folder_path	是	str	本地待上传文件夹的绝对路径
path_in_repo	否	str	文件夹将被上传到的具体路径及设置的文件夹名称
commit_message	否	str	此次上传提交所包含的更改内容
token	否	str	有权限上传的用户访问令牌。前置已经登陆时，可缺省
repo_type	否	str	仓库类型：model, dataset,不填默认为model
allow_patterns	否	str	允许上传的文件类型模板，例如*.json， 默认为None
ignore_patterns	否	str	上传时忽略的文件类型模板，例如*.log，默认为None
max_workers	否	int	上传时开启的线程数量，默认为 min(8,os.cpu_count() + 4))
revision	否	str	上传的分支，默认为master
'''
owner_name = 'rickxt'
dataset_name = 'nanochatd4'

api.upload_folder(
    repo_id=f"{owner_name}/{dataset_name}",
    folder_path='base_checkpoints',
    path_in_repo='nanochatd4/base_checkpoints',
    commit_message='d4模型',
    repo_type = 'dataset'
)

api.upload_folder(
    repo_id=f"{owner_name}/{dataset_name}",
    folder_path='mid_checkpoints',
    path_in_repo='nanochatd4/mid_checkpoints',
    commit_message='d4模型',
    repo_type = 'dataset'
)

api.upload_folder(
    repo_id=f"{owner_name}/{dataset_name}",
    folder_path='chatsft_checkpoints',
    path_in_repo='nanochatd4/chatsft_checkpoints',
    commit_message='d4模型',
    repo_type = 'dataset'
)

api.upload_folder(
    repo_id=f"{owner_name}/{dataset_name}",
    folder_path='base_eval',
    path_in_repo='nanochatd4/base_eval',
    commit_message='d4模型',
    repo_type = 'dataset'
)

api.upload_folder(
    repo_id=f"{owner_name}/{dataset_name}",
    folder_path='eval_bundle',
    path_in_repo='nanochatd4/eval_bundle',
    commit_message='d4模型',
    repo_type = 'dataset'
)

api.upload_folder(
    repo_id=f"{owner_name}/{dataset_name}",
    folder_path='logs',
    path_in_repo='nanochatd4/logs',
    commit_message='d4模型',
    repo_type = 'dataset'
)

api.upload_folder(
    repo_id=f"{owner_name}/{dataset_name}",
    folder_path='report',
    path_in_repo='nanochatd4/report',
    commit_message='d4模型',
    repo_type = 'dataset'
)
