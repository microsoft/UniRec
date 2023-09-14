package com.example;
import ai.onnxruntime.*;
import ai.onnxruntime.OrtException;
  
import java.io.*;  
import java.nio.file.*;  
import java.util.*;

public class Main {
    public static int nUsers = 940;  
    public static int nItems = 1017;  
    public static int embeddingSize = 64;  
    public static int maxSeqLen = 10;  
    public static int batchSize = 512;  
    public static boolean hasFeature = false;  
    public static int nFeatures = 2;  
    public static List<String> usefulNames = Arrays.asList("item_id", "item_seq");  

    public static void main(String[] args) {
        String historyFile = "path/to/user_history.csv";  
        String featureFile = "path/to/item_features.csv";  
        String testFile = "path/to/test.csv";  
        String modelPath = "path/to/model.onnx";  
        String outputDir = "path/to/output";  
        String taskType = "score"; //score, user embedding, item embedding  
        predictOnce(modelPath, historyFile, featureFile, testFile, outputDir, taskType);  
    }
    public static void predictOnce(String modelPath, String historyFile, String featureFile, String testFile, String outputDir, String taskType) {  
        System.out.println("Loading data and model weights...");  
        DataNode.Data data = SeqRecHelper.loadData(historyFile, featureFile, testFile);  
        // List<String> outputNames = Arrays.asList("scores", "user_embedding", "item_embedding");  
        try (OrtEnvironment environment = OrtEnvironment.getEnvironment();  
            OrtSession session = environment.createSession(new File(modelPath).getAbsolutePath(), new OrtSession.SessionOptions())) {  
  
            System.out.println("Predicting...");  
            int testLength = data.testSet.size();
            System.out.println("testLength: " + testLength);
            float[][] userEmbOut = new float[testLength][Main.embeddingSize];  
            float[][] itemEmbOut = new float[testLength][Main.embeddingSize];
            float[] scoreOut = new float[testLength];  
            for (int start = 0; start < testLength; start += Main.batchSize) {  
                int end = Math.min(start + Main.batchSize, testLength);  
                System.out.println("Predicting " + start + " to " + end);  
                DataNode.BatchInput input = SeqRecHelper.collateBatch(data, start, end);  
  
                Map<String, OnnxTensor> inputMap = new HashMap<>();
                for (String Name : Main.usefulNames) {  
                    if (Name.equals("user_id")) {  
                        inputMap.put(Name, OnnxTensor.createTensor(environment, input.userId));  
                    } else if (Name.equals("item_id")) {  
                        inputMap.put(Name, OnnxTensor.createTensor(environment, input.itemId));  
                    } else if (Name.equals("item_features")) {  
                        inputMap.put(Name, OnnxTensor.createTensor(environment, input.itemFeatures));  
                    } else if (Name.equals("item_seq")) {  
                        inputMap.put(Name, OnnxTensor.createTensor(environment, input.itemSeq));  
                    } else if (Name.equals("item_seq_len")) {  
                        inputMap.put(Name, OnnxTensor.createTensor(environment, input.itemSeqLen));  
                    } else if (Name.equals("item_seq_features")) {  
                        inputMap.put(Name, OnnxTensor.createTensor(environment, input.itemSeqFeatures));  
                    } else if (Name.equals("time_seq")) {  
                        inputMap.put(Name, OnnxTensor.createTensor(environment, input.timeSeq));  
                    } else {  
                        System.out.println("Wrong input name!");  
                    }  
                }   

                try (OrtSession.Result results = session.run(inputMap)) {
                    float[] batch_scoreOut = (float[]) results.get(0).getValue();
                    float[][] batch_userEmbOut = (float[][]) results.get(1).getValue();
                    float[][] batch_itemEmbOut = (float[][]) results.get(2).getValue();
                    for (int i = 0; i < end - start; i++) {  
                        scoreOut[start + i] = batch_scoreOut[i];  
                        for (int j = 0; j < Main.embeddingSize; j++) {  
                            userEmbOut[start + i][j] = batch_userEmbOut[i][j];  
                            itemEmbOut[start + i][j] = batch_itemEmbOut[i][j];  
                        }  
                    }
                }  
            }  
  
            System.out.println("Writing results to file...");  
  
            if (taskType.equals("score")) {  
                String outputScoreFile = outputDir + "score.txt";  
                try (BufferedWriter bw = Files.newBufferedWriter(Paths.get(outputScoreFile))) {  
                    for (float score : scoreOut) {  
                        bw.write(Float.toString(score) + "\n");  
                    }  
                } catch (IOException e) {  
                    e.printStackTrace();  
                }  
            } else if (taskType.equals("user embedding")) {  
                String outputUserEmbFile = outputDir + "user_embedding.txt";  
                try (BufferedWriter bw = Files.newBufferedWriter(Paths.get(outputUserEmbFile))) {  
                    for (int i = 0; i < testLength; i++) {  
                       
                        for (int j = 0; j < Main.embeddingSize; j++) {  
                            bw.write(Float.toString(userEmbOut[i][j]) + " ");  
                        }  
                        bw.write("\n");  
                    }  
                } catch (IOException e) {  
                    e.printStackTrace();  
                }  
            } else if (taskType.equals("item embedding")) {  
                String outputItemEmbFile = outputDir + "item_embedding.txt";  
                try (BufferedWriter bw = Files.newBufferedWriter(Paths.get(outputItemEmbFile))) {  
                    for (int i = 0; i < testLength; i++) {  
                        for (int j = 0; j < Main.embeddingSize; j++) {  
                            bw.write(Float.toString(itemEmbOut[i][j]) + " ");  
                        }  
                        bw.write("\n");  
                    }  
                } catch (IOException e) {  
                    e.printStackTrace();  
                }  
            } else {  
                System.out.println("Wrong task type!");  
            }  
  
            System.out.println("Done!");  
        } catch (OrtException e) {  
            e.printStackTrace();  
        }
    }  
}