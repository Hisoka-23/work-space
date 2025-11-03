import java.util.*;

public class ABC{
	public static void main(String[] args){
		Scanner obj = new Scanner(System.in);
		System.out.println("Enter the number of row : ");
		int row = obj.nextInt();

		char ch = 'A';
		for(int i=1; i<=row; i++){
			for(int j=1; j<=i; j++ ){
				System.out.print(ch+" ");
				ch++;
			}
		System.out.println();
		}
		
	}
}