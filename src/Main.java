/*
 * MultiClass document classification 
 * implementation of one-vs-all method
 * learning and classifying
 * 
 */

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightInterface;
import jnisvmlight.SVMLightModel;
import jnisvmlight.TrainingParameters;


public class Main {
	
	private static int POLITICS_LABEL = 4;
	private static int SPORTS_LABEL = 2;
	private static int TECH_LABEL = 3;
	private static int GENERAL_LABEL = 1;
	
	private static String SPORTS_MODEL_FILENAME = "Models//sportsModel";
	private static String POLITICS_MODEL_FILENAME = "Models//politicsModel";
	private static String TECH_MODEL_FILENAME = "Models//techModel";
	static String FEATURES_FILENAME = "featureToIndex.txt";
	
	public final static String ARABIC_CHARACHTERS = "اأإآبتثجحخدذرزسشصضطظعغفقكلمنهويىئءؤة";
	public final static String NON_ARABIC_CHARACHTER = "[^" + ARABIC_CHARACHTERS + "]";
	
	public static double THRESHOLD = 0;

	public static void main(String[] args) throws IOException {
		
		// read Arabic stop words
		ArrayList<String> stopWords = readStopWords();
		
		// training documents
		String politicsfoldername = "LearningDataSet\\politics";
		String sportsfoldername = "LearningDataSet\\sports";
		String techfoldername = "LearningDataSet\\tech";
		
		ArrayList<String> politicstrainingDocs = readDocuments(politicsfoldername);
		ArrayList<String> sportstrainingDocs = readDocuments(sportsfoldername);
		ArrayList<String> techtrainingDocs = readDocuments(techfoldername);

		
		// ========================= LEARNING ===================================
		// features
		HashMap<Integer, String> indexToFeature = new HashMap<Integer, String>();
		HashMap<String, Integer> featureToIndex = new HashMap<String, Integer>();
		ArrayList<String> allTrainingDocuments = addAllDocuments(politicstrainingDocs,
				sportstrainingDocs, techtrainingDocs);
		getFeatures(allTrainingDocuments, stopWords, indexToFeature, featureToIndex);
		
		// save featureToIndex.
		saveFeatureToIndex(featureToIndex);

		// using one-vs-all method
		// build sports model
		SVMLightModel sportsModel = buildSportsModel(sportstrainingDocs, politicstrainingDocs, techtrainingDocs
				, featureToIndex);
		sportsModel.writeModelToFile(SPORTS_MODEL_FILENAME);
		System.out.println("sports model file created");
		
		// build politics model
		SVMLightModel politicsModel = buildPoliticsModel(sportstrainingDocs, politicstrainingDocs, techtrainingDocs
					, featureToIndex);
		politicsModel.writeModelToFile(POLITICS_MODEL_FILENAME);
		System.out.println("politics model file created");
		 
		// build technology model
		 SVMLightModel techModel = buildTechModel(sportstrainingDocs, politicstrainingDocs, techtrainingDocs
				, featureToIndex);
		 techModel.writeModelToFile(TECH_MODEL_FILENAME);
		System.out.println("tech model file created");
		
		
		// ========================= EVALUATION ===================================
		// evaluating test document
		/*ArrayList<String> testDocs = readDocuments("TestingData");
		
		for(String doc : testDocs) {
			double politicsScore = classifyDocument(doc, politicsModel, featureToIndex);
			double sportsScore = classifyDocument(doc, sportsModel, featureToIndex);
			double techScore = classifyDocument(doc, techModel, featureToIndex);
			int res = detectClass(politicsScore, sportsScore, techScore);
			String resStr = "";
			switch(res) {
			case 1:
				resStr = "politics";
				break;
			case 2:
				resStr = "sports";
				break;
			case 3:
				resStr = "technology";
				break;
			case 4:
				resStr = "general";
				break;
			}
			
			System.out.println(doc.substring(0, 100));
			
			System.out.println("=========> Classified as " + resStr);
			System.out.println("-----------------------------------------------------");
		}*/
		
		// evaluating all documents in test folders	
		
		ArrayList<String> politicsTestDocuments = readDocuments("TestingDataSet\\politics");
		ArrayList<String> sportsTestDocuments = readDocuments("TestingDataSet\\sports");
		ArrayList<String> techTestDocuments = readDocuments("TestingDataSet\\tech");
		
		int total = 0;
		int right = 0;
		
		System.out.println("***************** POLITICS ***************************");
		for(String doc : politicsTestDocuments) {
			double politicsScore = classifyDocument(doc, politicsModel, featureToIndex);
			double sportsScore = classifyDocument(doc, sportsModel, featureToIndex);
			double techScore = classifyDocument(doc, techModel, featureToIndex);
			int res = detectClass(politicsScore, sportsScore, techScore);
			if(res == POLITICS_LABEL)
				right++;
			total++;
		}
		System.out.println("***************** SPORTS ***************************");
		for(String doc : sportsTestDocuments) {
			double politicsScore = classifyDocument(doc, politicsModel, featureToIndex);
			double sportsScore = classifyDocument(doc, sportsModel, featureToIndex);
			double techScore = classifyDocument(doc, techModel, featureToIndex);
			int res = detectClass(politicsScore, sportsScore, techScore);
			if(res == SPORTS_LABEL)
				right++;
			total++;
			
		}
		System.out.println("***************** TECHNOLOGY ***************************");
		for(String doc : techTestDocuments) {
			double politicsScore = classifyDocument(doc, politicsModel, featureToIndex);
			double sportsScore = classifyDocument(doc, sportsModel, featureToIndex);
			double techScore = classifyDocument(doc, techModel, featureToIndex);
			int res = detectClass(politicsScore, sportsScore, techScore);
			if(res == TECH_LABEL)
				right++;
			total++;
			
			System.out.println("score from politics model: " +
					politicsScore);
			System.out.println("score from sports model: " +
					sportsScore);
			System.out.println("score from tech model: " +
					techScore);
		}
		
		System.out.println("accuracy: " + ((double) right / (double) total) * 100);
		
	}
	
	private static int detectClass(double politicsScore, double sportsScore, double techScore) {
		
		if(politicsScore >= sportsScore && politicsScore >= techScore) {
			if(politicsScore > THRESHOLD)
				return POLITICS_LABEL;
			
		}
		else if(sportsScore >= politicsScore && sportsScore >= techScore) {
			if(sportsScore >= THRESHOLD)
				return SPORTS_LABEL;
		}
		else {
			if(techScore > THRESHOLD)
				return TECH_LABEL;
		}
		return GENERAL_LABEL;
	}
	
	private static double classifyDocument(String doc, SVMLightModel svmModel,
			HashMap<String, Integer> featureToIndex) {
		int anyLabel = 0; // not important
		LabeledFeatureVector featureVector = createFeatureVector(doc, featureToIndex, anyLabel);
		if(featureVector == null){
			System.out.println("cannot get feature vector from document");
			System.out.println(doc);
			return 0.0;
		}
		double score = svmModel.classify(featureVector);
		return score;
	}
	
	private static SVMLightModel buildTechModel(ArrayList<String> sportsTrainingDocs, ArrayList<String> politicsTrainingDocs
			, ArrayList<String> techTrainingDocs, HashMap<String, Integer> featureToIndex) {
		
		ArrayList<String> otherDocs = addAllDocuments(sportsTrainingDocs, politicsTrainingDocs);
		
		ArrayList<LabeledFeatureVector> trainingVectors = new ArrayList<>();
		
		for(String doc : techTrainingDocs) {
			LabeledFeatureVector featureVector = createFeatureVector(doc, featureToIndex, 1);
			if (featureVector != null) 
				trainingVectors.add(featureVector);
		}
		
		for(String doc : otherDocs) {
			LabeledFeatureVector featureVector = createFeatureVector(doc, featureToIndex, -1);
			if(featureVector != null)
				trainingVectors.add(featureVector);
		}
		
		SVMLightModel model = buildSVMModel(trainingVectors);
		model.writeModelToFile(TECH_MODEL_FILENAME);
		
		return model;
	}
	
	private static SVMLightModel buildPoliticsModel(ArrayList<String> sportsTrainingDocs, ArrayList<String> politicsTrainingDocs
			, ArrayList<String> techTrainingDocs, HashMap<String, Integer> featureToIndex) {
		
		ArrayList<String> otherDocs = addAllDocuments(sportsTrainingDocs, techTrainingDocs);
		
		ArrayList<LabeledFeatureVector> trainingVectors = new ArrayList<>();
		
		for(String doc : politicsTrainingDocs) {
			LabeledFeatureVector featureVector = createFeatureVector(doc, featureToIndex, 1);
			if (featureVector != null) 
				trainingVectors.add(featureVector);
		}
		
		for(String doc : otherDocs) {
			LabeledFeatureVector featureVector = createFeatureVector(doc, featureToIndex, -1);
			if(featureVector != null)
				trainingVectors.add(featureVector);
		}
		
		SVMLightModel model = buildSVMModel(trainingVectors);
		model.writeModelToFile(POLITICS_MODEL_FILENAME);
		return model;
	}
	
	private static void saveFeatureToIndex(HashMap<String, Integer> featureToIndex) throws FileNotFoundException, UnsupportedEncodingException {
		
		PrintWriter writer = new PrintWriter(FEATURES_FILENAME, "UTF-8");
		
		for(String key : featureToIndex.keySet()) {
			writer.println(key + " " + featureToIndex.get(key));
		}
		
		writer.close();
	}
	
	private static SVMLightModel buildSportsModel(ArrayList<String> sportsTrainingDocs, ArrayList<String> politicsTrainingDocs
			, ArrayList<String> techTrainingDocs, HashMap<String, Integer> featureToIndex) {
		
		ArrayList<String> otherDocs = addAllDocuments(politicsTrainingDocs, techTrainingDocs);
		
		ArrayList<LabeledFeatureVector> trainingVectors = new ArrayList<>();
		
		for(String doc : sportsTrainingDocs) {
			LabeledFeatureVector featureVector = createFeatureVector(doc, featureToIndex, 1);
			if (featureVector != null) 
				trainingVectors.add(featureVector);
		}
		
		for(String doc : otherDocs) {
			LabeledFeatureVector featureVector = createFeatureVector(doc, featureToIndex, -1);
			if(featureVector != null)
				trainingVectors.add(featureVector);
		}
		
		SVMLightModel model = buildSVMModel(trainingVectors);
		model.writeModelToFile(SPORTS_MODEL_FILENAME);
		return model;
	}
	
	private static SVMLightModel buildSVMModel(ArrayList<LabeledFeatureVector> trainingVectors) {

		SVMLightInterface trainer = new SVMLightInterface();
		SVMLightInterface.SORT_INPUT_VECTORS = true;
		LabeledFeatureVector[] traindataArray = new LabeledFeatureVector[trainingVectors.size()];
		traindataArray = trainingVectors.toArray(traindataArray);
		TrainingParameters tp = new TrainingParameters();

		// Switch on some debugging output
		tp.getLearningParameters().verbosity = 1;
		//System.out.println("\nTRAINING SVM-light MODEL ..");
		SVMLightModel model = trainer.trainModel(traindataArray, tp);
		//System.out.println("model" + model.toString());
		//System.out.println(" DONE Building Model.");

		return model;
	}
	
	/*
	private static ArrayList<LabeledFeatureVector> constructTrainingVectors(ArrayList<String> politicsTrainingDocs,
			ArrayList<String> sportsTrainingDocs, ArrayList<String> techTrainingDocs,
			ArrayList<String> stopWords, HashMap<String, Integer> featureToIndex) {

		ArrayList<LabeledFeatureVector> trainingVectors = new ArrayList<>();

		for (String doc : politicsTrainingDocs) {
			LabeledFeatureVector featureVector = createFeatureVector(doc, featureToIndex, POLITICS_LABEL);
			if (featureVector != null) {
				trainingVectors.add(featureVector);
			}
		}

		for (String doc : sportsTrainingDocs) {
			LabeledFeatureVector featureVector = createFeatureVector(doc, featureToIndex, SPORTS_LABEL);
			if (featureVector != null) {
				trainingVectors.add(featureVector);
			}
		}
		
		for (String doc : techTrainingDocs) {
			LabeledFeatureVector featureVector = createFeatureVector(doc, featureToIndex, TECH_LABEL);
			if (featureVector != null) {
				trainingVectors.add(featureVector);
			}
		}

		return trainingVectors;
	}*/
	

	private static LabeledFeatureVector createFeatureVector(String doc, HashMap<String, Integer> featureToIndex,
			int label) {

		String text = doc;
		text = text.trim();
		if (text.length() == 0)
			return null;

		String[] tokens = text.split(" +");
		if (tokens.length == 0)
			return null;

		ArrayList<Integer> documentFeatures = getfeaturesFromDocument(featureToIndex, tokens);
		LabeledFeatureVector labeledFeatureVector = createLabeledFeatureVector(documentFeatures, label);
		return labeledFeatureVector;
	}
	
	private static ArrayList<Integer> getfeaturesFromDocument(HashMap<String, Integer> featuresMap, String[] tokens) {

		ArrayList<Integer> documentFeatures = new ArrayList<Integer>();
		for (String token : tokens) {
			token = token.trim();
			if (token.length() == 0)
				continue;

			Integer index = featuresMap.get(token);
			if (index != null)
				if (!documentFeatures.contains(index))
					documentFeatures.add(index);

		}
		Collections.sort(documentFeatures);
		
		return documentFeatures;
	}

	private static LabeledFeatureVector createLabeledFeatureVector(ArrayList<Integer> documentFeatures, int label) {
		int nDims = documentFeatures.size();
		int[] dims = new int[nDims];
		double[] values = new double[nDims];

		int i = 0;
		for (; i < documentFeatures.size(); i++) {
			dims[i] = documentFeatures.get(i);
			values[i] = 1.0; // could be term frequency or tf-idf
		}

		LabeledFeatureVector labelFeatureVector = new LabeledFeatureVector(label, dims, values);
		labelFeatureVector.normalizeL2();
		return labelFeatureVector;
	}

	
	private static void getFeatures(ArrayList<String> trainingDocs, ArrayList<String> stopWords,
			HashMap<Integer, String> indexToFeature, HashMap<String, Integer> featureToIndex) {

		int index = 1;
		for (String doc : trainingDocs) {
			String[] tokens = doc.split(" +");
			for (String token : tokens) {
				if (stopWords.contains(token)|| featureToIndex.containsKey(token) ||
						token == null || token.isEmpty())
					continue;
				indexToFeature.put(index, token);
				featureToIndex.put(token, index);
				index++;
			}
		}

	}

	private static ArrayList<String> addAllDocuments(ArrayList<String> politicstrainingDocs,
			ArrayList<String> sportstrainingDocs, ArrayList<String> techtrainingDocs) {
		ArrayList<String> allTrainingDocuments = new ArrayList<>();
		allTrainingDocuments.addAll(politicstrainingDocs);
		allTrainingDocuments.addAll(sportstrainingDocs);
		allTrainingDocuments.addAll(techtrainingDocs);
		return allTrainingDocuments;
	}
	
	private static ArrayList<String> addAllDocuments(ArrayList<String> docs1, ArrayList<String> docs2) {
		ArrayList<String> allTrainingDocuments = new ArrayList<>();
		allTrainingDocuments.addAll(docs1);
		allTrainingDocuments.addAll(docs2);
		return allTrainingDocuments;
	}
		
	private static ArrayList<String> readDocuments(String foldername) throws IOException {
		ArrayList<String> trainingDocs = new ArrayList<>();

		File directory = new File(foldername);

		// get all the files from a directory
		File[] fList = directory.listFiles();
		for (File file : fList) {
			if (file.isFile()) {
				String filePath = foldername + "\\" + file.getName();
				trainingDocs.add(readDocument(filePath));
			}
		}

		return trainingDocs;
	}
	
	private static String readDocument(String fileName) throws IOException {

		String sCurrentLine;
		StringBuilder sb = new StringBuilder("");
		BufferedReader br = new BufferedReader(new FileReader(fileName));

		while ((sCurrentLine = br.readLine()) != null) {
			sb.append(sCurrentLine + " ");
		}
		br.close();

		return sb.toString().replaceAll(NON_ARABIC_CHARACHTER, " ");
	}
	
	// returns arabic stop words.
	private static ArrayList<String> readStopWords() throws IOException {
		
		ArrayList<String> stopWords = new ArrayList<String>();
		String sCurrentLine;
		BufferedReader br = new BufferedReader(new FileReader("StopWords\\stopwords.txt"));
		while ((sCurrentLine = br.readLine()) != null) {
			stopWords.add(sCurrentLine.trim());
		}
		
		br.close();
		
		return stopWords;
	}

}
