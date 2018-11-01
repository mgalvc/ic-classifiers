public class Runner {

    public void run() throws Exception {
        final ClassLoader classLoader = getClass().getClassLoader();
        String relation = "abalone";
        String datasetPath = "otm_datasets_small/NSGAIII_datasets/" + relation + "/";
        Classifiers cls = new Classifiers(classLoader.getResource(datasetPath).getPath(), relation);

        cls.run(true);
    }

    public static void main(String[] args) throws Exception {
        Runner runner = new Runner();
        runner.run();
    }
}
