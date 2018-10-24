public class Runner {

    public void run() throws Exception {
        final ClassLoader classLoader = getClass().getClassLoader();
        String relation = "marketing";
        String datasetPath = "datasets_small/" + relation + "/";
        Classifiers cls = new Classifiers(classLoader.getResource(datasetPath).getPath(), relation);

        cls.run(false);
    }

    public static void main(String[] args) throws Exception {
        Runner runner = new Runner();
        runner.run();
    }
}
