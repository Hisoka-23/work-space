package in.prakash.config;

import org.springframework.batch.item.ItemProcessor;
import org.springframework.stereotype.Component;

import in.prakash.entity.Customer;

@Component
public class CustomerProcessor implements ItemProcessor<Customer, Customer>{

	public Customer process(Customer item) throws Exception{
		
		//logic to process data
		/*
		 * if(item.getCountry().equals("India")) { return item; } else { return null; }
		 */
		
		 return item; 
		 
	}
	
}
