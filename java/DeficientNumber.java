import java.util.*;

public class DeficientNumber {
	public static void main(String[] args){
		Scanner obj = new Scanner(System.in);
		System.out.println("Enter the number :: ");
		int num = obj.nextInt();
	
		int sum = 0;

		for(int i=1; i<num; i++){
			if(num % i == 0){
				sum = sum + i;
			}
		}

		if(sum < num){
			System.out.println("this is DeficientNumber ");
		} else {
			System.out.println("this is not DeficientNumber ");
		}
	}
}