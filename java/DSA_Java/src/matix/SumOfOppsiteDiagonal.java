package matix;

import java.util.Scanner;

public class SumOfOppsiteDiagonal {

	public static void main(String[] args) {
		Scanner obj = new Scanner(System.in);

		System.out.println("Enter row value: ");
		int row = obj.nextInt();
		
		System.out.println("Enter col value: ");
		int col = obj.nextInt();
		
		int a[][] = new int[row][col];
		
		int i,j;
		
		System.out.println("Enter matrix element : ");
		for(i=0; i<row; i++) {
			for(j=0; j<col; j++) {
				a[i][j] = obj.nextInt();
			}
		}
		
		System.out.println("Matrix : ");
		for(i=0; i<row; i++) {
			for(j=0; j<col; j++) {
				System.out.print(a[i][j]+" ");
			}
			System.out.println();
		}
		
		int sum = 0;
		

		for(i=0; i<row; i++) {
					sum = sum + a[i][row - i -1];
		}
		
		System.out.println("Some of Daigonal : "+ sum);
	}
	
	
}
