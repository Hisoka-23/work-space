package Recursion;

import java.util.Scanner;

//03. Implement a program to calculate a^b (a to the power b)
class Demo3{
	static int power(int a, int b) {
		if(b>=1) {
			return a*power(a, b-1);
		} else {
			return 1;
		}
	}
}
public class AtoThePowerB {
	
	public static void main(String[] args) {
		Scanner obj1 = new Scanner(System.in);
		Scanner obj2 = new Scanner(System.in);
		
		int a = obj1.nextInt();
		int b =  obj2.nextInt();
		
		System.out.println(Demo3.power(a, b));
	}

}
