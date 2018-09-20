public class Runner {

    public void run() throws Exception {
        final ClassLoader classLoader = getClass().getClassLoader();
        String datasetPath = "otm_datasets_small/NSGADO_datasets/abalone/";
        Classifiers cls = new Classifiers(classLoader.getResource(datasetPath).getPath(), "abalone");

        cls.run();
    }

    public static void main(String[] args) throws Exception {
        Runner runner = new Runner();
        runner.run();
    }
}
