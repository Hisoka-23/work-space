package Recursion;

import java.util.Scanner;

class Demo9{
	static boolean isprime(int n, int i) {
		if(i==1) {
			return true;
		} else if(n%i==0) {
			return false;
		} else {
			return isprime(n, --i);
		}
	}
}

public class PrimeCheck {

	public static void main(String[] args) {
		
		Scanner obj = new Scanner(System.in);
		
		System.out.println("Enter n value...");
		
		int n = obj.nextInt();
		
		System.out.println(Demo9.isprime(n, n/2));
		
	}
	
}
