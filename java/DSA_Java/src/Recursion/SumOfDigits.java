package Recursion;

import java.util.Scanner;

class Demo10{
	static int sumOfDigits(int n) {
		if(n == 0) {
			return 0;
		} else {
			return n%10+sumOfDigits(n/10);
		}
	}
}

public class SumOfDigits {

	public static void main(String[] args) {
		
		Scanner obj = new Scanner(System.in);
		
		System.out.println("Enter n value...!");
		
		int n = obj.nextInt();
		
		System.out.println(Demo10.sumOfDigits(n));
		
	}
	
}
