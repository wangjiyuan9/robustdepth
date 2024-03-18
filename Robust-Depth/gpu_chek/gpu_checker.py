#coding=utf-8
import os
import sys
import subprocess
import smtplib
import time

#要执行的命令文件路径
COMMAND_FILE_PATH = "wait2run.txt"
# 程序执行前需要进入的路径
PROGRAM_PATHS = {
    1: "/opt/data/private/wjy/PlaneDepth/",
    10: "/data/cylin/wjy/wavelet-monodepth/"
}
# 程序使用的环境名
CONDA_ENV_NAME = "wavelet-wjy"
# 邮箱信息
MAIL_SERVER = "smtp.qq.com"
MAIL_PORT = 465
MAIL_FROM = "2939777532@qq.com"
MAIL_PASSWORD = "zzexzwezjjzwdggb"
MAIL_TO = "wangjiyuan5@163.com"
MAIL_SUBJECT = "GPU程序开始执行"
MAIL_CONTENT_TEMPLATE = "已执行：{command}，在{gpu}卡上"


def send_mail(content):
    """发送邮件"""
    with smtplib.SMTP_SSL(MAIL_SERVER, MAIL_PORT) as smtp:
        smtp.login(MAIL_FROM, MAIL_PASSWORD)
        message = "From: {}\nTo: {}\nSubject: {}\n\n{}".format(MAIL_FROM, MAIL_TO, MAIL_SUBJECT, content)
        smtp.sendmail(MAIL_FROM, MAIL_TO, message.encode("utf-8"))

def get_gpu_info():
    cmd = 'nvidia-smi'
    gpu_info = subprocess.check_output(cmd, shell=True).decode().strip().split('\n')
    return gpu_info

def get_gpu_usage():
    """
    获取GPU使用情况
    """
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"])
    return result.decode('utf-8').strip()


def get_available_gpus():
    """
    获取可用的GPU
    """
    gpu_info = get_gpu_usage().split("\n")
    available_gpus = []
    for gpu in gpu_info:
        gpu_stat = gpu.split(",")
        gpu_memory = int(gpu_stat[2])
        gpu_power = int(gpu_stat[1])
        if gpu_memory < 1000 and gpu_power < 20:
            available_gpus.append(int(gpu_stat[0]))
    return available_gpus, len(gpu_info)


def execute_command(command, gpu, gpu_num):
    """
    执行命令，并发送邮件
    """
    program_path = PROGRAM_PATHS.get(gpu_num)
    python_path = sys.executable
    # 获取model_name拼接增加的部分
    model_name = command.split("--model_name ")[1].split(" ")[0]
    command += " > result/{}.txt 2>&1 &".format(model_name)
    if program_path:
        command = "cd {} && {} ".format(program_path, command)
    command = command.replace("{}", str(gpu)).replace("python", python_path)
    os.system(command)
    # result = subprocess.Popen(['bash', '-c', command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # print(result.stdout)
    # print(result.stderr)
    # send_mail(MAIL_CONTENT_TEMPLATE.format(command=command, gpu=gpu))


if __name__ == "__main__":
    while(True):
        # 读取命令文件
        with open(COMMAND_FILE_PATH, "r") as f:
            commands = f.read().split("\n")
        gpu_num = 0
        commands = [command for command in commands if command.strip() != ""]
        if len(commands) == 0:
            time.sleep(60)
            continue
        for command in commands:
            # 获取可用的GPU
            available_gpus, gpu_num = get_available_gpus()
            # 如果没有可用的GPU，等待10秒后重试
            while not available_gpus:
                time.sleep(60)
                available_gpus, gpu_num = get_available_gpus()
            # 执行命令，并发送邮件
            execute_command(command.strip(), available_gpus[0], gpu_num)
            print("command executed:", command,"Now Time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),"in gpu:",available_gpus[0])
            # 删除已执行的命令
            with open(COMMAND_FILE_PATH, "r") as f:
                lines = f.readlines()
            with open(COMMAND_FILE_PATH, "w") as f:
                for line in lines:
                    if line.strip() != command.strip():
                        f.write(line)
            #等待程序全面启动
            time.sleep(60)
        # 文件所有命令执行完成后，等待1小时后再次检查有没有新加入的命令
        print("all commands executed, wait 1 minute to check again")