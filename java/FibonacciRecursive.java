import java.util.*;

public class FibonacciRecursive {

static int Fibonacci(int n){
	if(n == 0){
		return 0;
	} else if(n == 1){
		return 1;
	} else {
		return  Fibonacci(n-1) + Fibonacci(n-2);
	}
}

	public static void main(String[] args){
		Scanner obj = new Scanner(System.in);
		System.out.println("Enter the number : ");
		int num = obj.nextInt();
		
		for(int i=0; i<num; i++){
			System.out.print(Fibonacci(i)+ " ");
		}
	}
}