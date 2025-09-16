package Recursion;

import java.util.Scanner;

//02. Implement a program to calculate sum of 'n' natural numbers

class Demo2{
	static int sum(int n) {
		if(n==1) 
			return 1;
		 else 
			return n+sum(n-1);
	}
}

public class SumOfNaturalNumber {
	
	public static void main(String[] args) {
		
		Scanner obj = new Scanner(System.in);
		
		int n = obj.nextInt();
		
		System.out.println(Demo2.sum(n));
	}

}
