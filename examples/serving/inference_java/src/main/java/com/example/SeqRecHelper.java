// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

package com.example;
import java.io.BufferedReader;  
import java.io.FileReader;  
import java.io.IOException;  
import java.util.*;  
  
public class SeqRecHelper {  
  
    public static DataNode.Data loadData(String historyFile, String featureFile, String testFile) {  
        DataNode.Data data = new DataNode.Data();  
  
        try (BufferedReader br = new BufferedReader(new FileReader(historyFile))) {  
            String line;  
            line = br.readLine();
            while ((line = br.readLine()) != null) {  
                DataNode.HistoryNode historyNode = new DataNode.HistoryNode();  
                String[] list = line.trim().split("\t");  
                historyNode.userId = Long.parseLong(list[0].trim());  
                historyNode.itemSeq = Arrays.stream(list[1].trim().split(",")).mapToLong(Long::parseLong).toArray();  
                if (list.length == 3) {  
                    historyNode.hasTime = true;  
                    historyNode.timeSeq = Arrays.stream(list[2].trim().split(",")).mapToLong(Long::parseLong).toArray();
                } else {  
                    historyNode.hasTime = false;  
                }  
                data.userHistory[(int) historyNode.userId] = historyNode;  
            }  
        } catch (IOException e) {  
            e.printStackTrace();  
        }  
  
        if (Main.hasFeature) {  
            data.itemFeatures = new DataNode.FeatureNode[Main.nItems];  
            data.itemFeatures[0] = new DataNode.FeatureNode();  
            try (BufferedReader br = new BufferedReader(new FileReader(featureFile))) {  
                String line;  
                line = br.readLine();
                while ((line = br.readLine()) != null) {  
                    DataNode.FeatureNode featureNode = new DataNode.FeatureNode();  
                    String[] list = line.trim().split("\t");  
                    featureNode.itemId = Long.parseLong(list[0].trim());  
                    featureNode.itemFeats = Arrays.stream(list[1].trim().split(",")).mapToLong(Long::parseLong).toArray();  
                    data.itemFeatures[(int) featureNode.itemId] = featureNode;  
                }  
            } catch (IOException e) {  
                e.printStackTrace();  
            }  
        }  
  
        data.testSet = new ArrayList<>();  
        try (BufferedReader br = new BufferedReader(new FileReader(testFile))) {  
            String line;  
            line = br.readLine();
            while ((line = br.readLine()) != null) {  
                DataNode.TestNode testNode = new DataNode.TestNode();  
                String[] list = line.trim().split("\t");  
                testNode.userId = Long.parseLong(list[0].trim());  
                testNode.itemId = Long.parseLong(list[1].trim());  
                data.testSet.add(testNode);  
            }  
        } catch (IOException e) {  
            e.printStackTrace();  
        }  
  
        return data;  
    }  
  
    public static DataNode.BatchInput collateBatch(DataNode.Data data, int l, int r) {  
        DataNode.BatchInput batchInput = new DataNode.BatchInput(r - l);  
        for (int i = l; i < r; i++) {  
            batchInput.userId[i - l] = data.testSet.get(i).userId;  
            batchInput.itemId[i - l] = data.testSet.get(i).itemId;  
            DataNode.HistoryNode history = data.userHistory[(int) data.testSet.get(i).userId];  
            long itemId = data.testSet.get(i).itemId;  
            int historyLen = history.itemSeq.length - 1;  
            for (; historyLen >= 0; historyLen--) {  
                if (history.itemSeq[historyLen] == itemId) {  
                    break;  
                }  
            }  
            if (historyLen == -1) {  
                historyLen = history.itemSeq.length;
            }
            batchInput.itemSeqLen[i - l] = Math.min(historyLen, Main.maxSeqLen);  
            for (int j = historyLen - 1, k = Main.maxSeqLen - 1; j >= Math.max(0, historyLen - Main.maxSeqLen); j--, k--) {  
                batchInput.itemSeq[i - l][k] = history.itemSeq[j];  
                if (history.hasTime) {  
                    batchInput.timeSeq[i - l][k] = history.timeSeq[j];  
                }  
            }  
        }  
        if (Main.hasFeature) {  
            batchInput.itemFeatures = new long[r - l][Main.nFeatures];  
            for (int i = l; i < r; i++) {  
                for (int j = 0; j < Main.nFeatures; j++) {  
                    batchInput.itemFeatures[i - l][j] = data.itemFeatures[(int) data.testSet.get(i).itemId].itemFeats[j];  
                }  
            }  
            batchInput.itemSeqFeatures = new long[r - l][Main.maxSeqLen][Main.nFeatures];  
            for (int i = l; i < r; i++) {  
                for (int j = 0; j < Main.maxSeqLen; j++) {  
                    for (int k = 0; k < Main.nFeatures; k++) {  
                        batchInput.itemSeqFeatures[i - l][j][k] = data.itemFeatures[(int) batchInput.itemSeq[i - l][j]].itemFeats[k];  
                    }  
                }  
            }  
        }  
        return batchInput;  
    }  
}  
