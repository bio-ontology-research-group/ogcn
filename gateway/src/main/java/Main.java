
import com.beust.jcommander.JCommander;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.ogcn.parse.*;

public class Main {

    Logger logger;


    public Main() {
	logger = LoggerFactory.getLogger(Main.class);
    }

    public void run() throws Exception{
	logger.info("Run function is excecuted");
    }

    public static void main(String[] args) {
    
	Main main = new Main();
	JCommander jcom = JCommander.newBuilder()
            .addObject(main)
            .build();
	try {
	    jcom.parse(args);
	    main.run();
	} catch (Exception e) {
	    e.printStackTrace();
	    jcom.usage();
	}
    }
}