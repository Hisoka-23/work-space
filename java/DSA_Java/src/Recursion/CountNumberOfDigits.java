package Recursion;

import java.util.Scanner;

//Implement a program to count number of digits present in the given number

class Demo7{
	
	static int c = 0;
	
	static int count(long n) {
		if(n != 0) {
			c++;
			count(n/10);
		}
		
		return (c!=0)?c:1;
	}
}

public class CountNumberOfDigits {

	public static void main(String[] args) {
		Scanner obj = new Scanner(System.in);
		long n = obj.nextLong();
		System.out.println(Demo7.count(n));
	}
	
}
