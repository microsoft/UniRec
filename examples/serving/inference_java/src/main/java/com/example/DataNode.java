package com.example;
import java.util.List;  
  
public class DataNode {  
    public static class HistoryNode {  
        public long userId;  
        public long[] itemSeq;  
        public long[] timeSeq;  
        public boolean hasTime;  
    }  
  
    public static class FeatureNode {  
        public long itemId;  
        public long[] itemFeats;  
  
        public FeatureNode() {  
            itemId = 0;  
            itemFeats = new long[Main.nFeatures];  
            for (int i = 0; i < Main.nFeatures; i++) {  
                itemFeats[i] = 0;  
            }  
        }  
    }  
  
    public static class TestNode {  
        public long userId;  
        public long itemId;  
    }  
  
    public static class Data {  
        public HistoryNode[] userHistory;  
        public FeatureNode[] itemFeatures;  
        public List<TestNode> testSet;  
  
        public Data() {  
            userHistory = new HistoryNode[Main.nUsers];  
        }  
    }  
  
    public static class BatchInput {  
        public long[] userId;  
        public long[] itemId;  
        public long[][] itemFeatures;  
        public long[][] itemSeq;  
        public long[] itemSeqLen;  
        public long[][][] itemSeqFeatures;  
        public long[][] timeSeq;  
  
        public BatchInput(int batchSize) {  
            userId = new long[batchSize];  
            itemId = new long[batchSize];  
            itemSeq = new long[batchSize][Main.maxSeqLen];  
            itemSeqLen = new long[batchSize];  
            timeSeq = new long[batchSize][Main.maxSeqLen];
            for (int i = 0; i < batchSize; i++) {  
                for (int j = 0; j < Main.maxSeqLen; j++) {  
                    itemSeq[i][j] = 0;  
                    timeSeq[i][j] = 0;  
                }
                itemSeqLen[i] = 0;
            }  
        }  
    }  
}