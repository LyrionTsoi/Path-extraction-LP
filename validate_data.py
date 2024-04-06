from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner
import pickle
import sys

        
def read_fasta_file(filepath):
    """读取FASTA格式的文件并返回序列记录列表"""
    sequences = list(SeqIO.parse(filepath, "fasta"))
    return sequences


def align_sequences(test_seq, ref_seqs):
    """对每个测试序列，找到与其局部对齐得分最高的参考序列"""
    aligner = PairwiseAligner()
    aligner.mode = 'local'  # 设置为局部对齐
    best_scores = []
    for test in test_seq:
        best_score = 0
        best_ref = None
        for ref in ref_seqs:
            score = aligner.align(test.seq, ref.seq).score
            if score > best_score:
                best_score = score
                best_ref = ref.id
        best_scores.append((test.id, best_ref, best_score))
    return best_scores

def align_sequence(test_seq, ref_seqs):
    """对每个测试序列，找到与其局部对齐得分最高的参考序列"""
    aligner = PairwiseAligner()
    aligner.mode = 'local'  # 设置为局部对齐
    best_scores = []
    for test in test_seq:
        best_score = 0
        best_ref = None
        for ref in ref_seqs:
            score = aligner.align(test.seq, ref.seq).score
            if score > best_score:
                best_score = score
                best_ref = ref.id
        best_scores.append((test.id, best_ref, best_score))
    return best_scores

def main():
    validation_filePath = "validation-set.fasta"
    original_filePath = "6-strain-polio.fasta"


    ref_seqs = read_fasta_file(original_filePath)
    test_seqs = read_fasta_file(validation_filePath)

    best_scores = align_sequences(test_seqs,ref_seqs)

    test_scores = {}
    # test_scores = align_sequences(ref_seqs, ref_seqs)

    # 将test_scores保存到文件
    # with open('test_scores.pkl', 'wb') as f:
    #     pickle.dump(test_scores, f)

    with open('test_scores.pkl', 'rb') as f:
        test_scores = pickle.load(f)

    # 首先，创建一个字典来存储每个参考序列自身比对的得分，以便快速访问
    ref_self_scores = {TID: TS for TID, TBR, TS in test_scores}

    # 然后，遍历best_scores列表，并用相应的参考序列得分来标准化得分
    normalized_scores = []
    for ID, best_ref, score in best_scores:
        # 获取参考序列自身的得分
        ref_score = ref_self_scores[best_ref]
        # 计算标准化得分
        normalized_score = score / ref_score
        # 存储更新后的得分
        normalized_scores.append((ID, best_ref, normalized_score))

    # 打印标准化后的得分
    # for ID, best_ref, score in normalized_scores:
    #     print(f"ID: {ID}, Best Ref: {best_ref}, Normalized Score: {score}")

    with open('validate-data.txt','a') as f:
        for ID, best_ref, score in normalized_scores:
            line = "ID: {ID}, Best Ref: {best_ref}, Normalized Score: {score}\n"
            f.write(line)
            

    # for ID, best_ref, score in best_scores:
    #     print(f"ID: {ID}")  # 序列的ID，比如seq_1, seq_2等
    #     print(f"Sequence: {str(best_ref)[:50]}...")  # 序列数据，只展示前50个字符以简化输出
    #     print("-" * 60)  # 打印分隔线
    #     print(f"{score}")

    # for ID, best_ref, score in test_scores:
    #     print(f"ID: {ID}")  # 序列的ID，比如seq_1, seq_2等
    #     print(f"Best refID: {best_ref}")  # 序列数据，只展示前50个字符以简化输出
    #     print("-" * 60)  # 打印分隔线
    #     print(f"{score}")


if __name__ == "__main__":
    main()