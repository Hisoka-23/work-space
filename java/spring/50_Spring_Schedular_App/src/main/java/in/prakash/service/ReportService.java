package in.prakash.service;

import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

@Service
public class ReportService {

	@Scheduled(cron = "0/3 * * * * *")//fixedDelay,fixedRate
	public void generateReport() {
		//logic
		System.out.println("Report generated...!!");
	}
	
}
