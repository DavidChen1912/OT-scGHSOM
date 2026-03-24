import os
import sys
import argparse
import csv
from programs.data_processing.format_ghsom_input_vector import format_ghsom_input_vector


# ============================================================
# ⭐ Pipeline Functions（原封不動，只更新傳參數）
# ============================================================
def create_ghsom_input_file(data, file, index, label, subnum):
    try:
        format_ghsom_input_vector(data, file, index, label, subnum)
        print('Success to create ghsom input file.')
    except Exception as e:
        print('Failed to create ghsom input file.')
        print('Error:', e)

def create_ghsom_prop_file(name, file, tau1=0.1, tau2=0.01,
                           sparseData='yes', isNormalized='false',
                           randomSeed=7, xSize=2, ySize=2,
                           learnRate=0.7, numIterations=20000):

    with open(f'./applications/{file}/GHSOM/{name}_ghsom.prop',
              'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(['workingDirectory=./'])
        writer.writerow([f'outputDirectory=./output/{file}'])
        writer.writerow([f'namePrefix={name}'])
        writer.writerow([f'vectorFileName=./data/{name}_ghsom.in'])
        writer.writerow([f'sparseData={sparseData}'])
        writer.writerow([f'isNormalized={isNormalized}'])
        writer.writerow([f'randomSeed={randomSeed}'])
        writer.writerow([f'xSize={xSize}'])
        writer.writerow([f'ySize={ySize}'])
        writer.writerow([f'learnRate={learnRate}'])
        writer.writerow([f'numIterations={numIterations}'])
        writer.writerow([f'tau={tau1}'])
        writer.writerow([f'tau2={tau2}'])

def ghsom_clustering(name, file):
    try:
        cmd = f'./programs/GHSOM/somtoolbox.sh GHSOM ./applications/{file}/GHSOM/{name}_ghsom.prop -h'
        print("cmd=", cmd)
        os.system(cmd)
    except Exception as e:
        print("Error:", e)

def extract_ghsom_output(name, current_path):
    cmd = f'7z e applications/{name}/GHSOM/output/{name} -o{current_path}/applications/{name}/GHSOM/output/{name}'
    print("cmd=", cmd)
    os.system(cmd)

def save_ghsom_cluster_label(name, tau1, tau2, index):
    cmd = f'python ./programs/data_processing/save_cluster_with_clustered_label.py --name={name} --tau1={tau1} --tau2={tau2} --index={index}'
    os.system(cmd)
    print('Success transfer cluster label.')

def clustering_evaluation(name, tau1=0.1, tau2=0.01, label=None, index=None):
    cmd = f'python ./programs/evaluation/clustering_scores.py --name={name} --tau1={tau1} --tau2={tau2}'
    
    if label is not None:
        cmd += f' --label={label}'

    if index is not None:
        cmd += f' --index={index}'

    os.system(cmd)
    print('Success evaluating.')


# ============================================================
# ⭐⭐ 封裝 Pipeline 主流程（模組化核心） ⭐⭐
# ============================================================
def run_pipeline(data, tau1, tau2, index=None, label=None, subnum=None, feature='mean'):
    """
    外部 scripts 也能呼叫：
    from execute import run_pipeline
    run_pipeline(data="xxx", tau1=0.08, tau2=0.2)
    """
    print(f"tau1 = {tau1}, tau2 = {tau2}")
    print(f"data = {data}, index = {index}, label = {label}")

    file = f"{data}-{tau1}-{tau2}"
    current_path = os.getcwd()
    print("Current:", current_path)

    # 建立 applications folder
    app_path = f'{current_path}/applications/{file}'
    if os.path.exists(app_path):
        print(f'Warning : /applications/{file} already exists.')
    else:
        print(f'Creating /applications/{file} ...')
        try:
            os.makedirs(f'{app_path}')
            os.makedirs(f'{app_path}/data')
            os.makedirs(f'{app_path}/GHSOM')
            os.makedirs(f'{app_path}/graphs')
            os.makedirs(f'{app_path}/GHSOM/data')
            os.makedirs(f'{app_path}/GHSOM/output')

            # Pipeline 順序（完全不變）
            create_ghsom_input_file(data, file, index, label, subnum)
            create_ghsom_prop_file(data, file, tau1, tau2)
            ghsom_clustering(data, file)
            extract_ghsom_output(file, current_path)
            save_ghsom_cluster_label(data, tau1, tau2, index)
            clustering_evaluation(data, tau1, tau2, label, index)

        except Exception as e:
            print(f'Failed to create /applications/{file} folder due to: {str(e)}')


# ============================================================
# ⭐ main(): 只有獨立執行時才會跑這裡
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='manual to this script')

    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--tau1', type=float, required=True)
    parser.add_argument('--tau2', type=float, required=True)

    parser.add_argument('--index', type=str, default=None)
    parser.add_argument('--label', type=str, default=None)

    parser.add_argument('--subnum', type=int, default=None)
    parser.add_argument('--feature', type=str, default='mean')

    args = parser.parse_args()

    run_pipeline(
        data=args.data,
        tau1=args.tau1,
        tau2=args.tau2,
        index=args.index,
        label=args.label,
        subnum=args.subnum,
        feature=args.feature
    )


# ============================================================
# ⭐ 入口點
# ============================================================
if __name__ == "__main__":
    main()


# python execute.py --data=Samusik_01_cleaned --index=Event --label=label --tau1=0.08 --tau2=0.2

