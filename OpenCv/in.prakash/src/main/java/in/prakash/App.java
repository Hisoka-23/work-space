package in.prakash;

/**
 * Hello world!
 *
 */
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class App extends Application {
    @Override
    public void start(Stage stage) throws Exception {
        FXMLLoader loader = new FXMLLoader(App.class.getResource("/fxml/main.fxml"));
        Scene scene = new Scene(loader.load(), 960, 720);
        stage.setTitle("JavaFX + JavaCV Face Recognition");
        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) {
        launch();
    }
}