package in.prakash.config;

import org.springframework.batch.core.Job;
import org.springframework.batch.core.Step;
import org.springframework.batch.core.job.builder.JobBuilder;
import org.springframework.batch.core.repository.JobRepository;
import org.springframework.batch.core.step.builder.StepBuilder;
import org.springframework.batch.item.data.RepositoryItemWriter;
import org.springframework.batch.item.file.FlatFileItemReader;
import org.springframework.batch.item.file.LineMapper;
import org.springframework.batch.item.file.mapping.BeanWrapperFieldSetMapper;
import org.springframework.batch.item.file.mapping.DefaultLineMapper;
import org.springframework.batch.item.file.transform.DelimitedLineTokenizer;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.FileSystemResource;
import org.springframework.transaction.PlatformTransactionManager;

import in.prakash.entity.Customer;
import in.prakash.repository.CustomerRepository;

@Configuration
public class BatchConfig {

    @Autowired
    private CustomerRepository customerRepo;

    @Autowired
    private JobRepository jobRepository;

    @Autowired
    private PlatformTransactionManager transactionManager;

    // Item Reader
    @Bean
    public FlatFileItemReader<Customer> customerReader() {
        FlatFileItemReader<Customer> itemReader = new FlatFileItemReader<>();
        itemReader.setResource(new FileSystemResource("src/main/resources/people_data_v2.csv"));
        itemReader.setName("customer-item-reader");
        itemReader.setLinesToSkip(1);
        itemReader.setLineMapper(lineMapper());
        return itemReader;
    }

    private LineMapper<Customer> lineMapper() {
        DefaultLineMapper<Customer> lineMapper = new DefaultLineMapper<>();

        DelimitedLineTokenizer tokenizer = new DelimitedLineTokenizer();
        tokenizer.setDelimiter(",");
        tokenizer.setStrict(false);
        tokenizer.setNames("id", "firstName", "lastName", "email", "gender", "contactNum", "country", "dob");

        BeanWrapperFieldSetMapper<Customer> mapper = new BeanWrapperFieldSetMapper<>();
        mapper.setTargetType(Customer.class);

        lineMapper.setLineTokenizer(tokenizer);
        lineMapper.setFieldSetMapper(mapper);
        return lineMapper;
    }

    // Processor
    @Bean
    public CustomerProcessor customerProcessor() {
        return new CustomerProcessor();
    }

    // Writer
    @Bean
    public RepositoryItemWriter<Customer> customerWriter() {
        RepositoryItemWriter<Customer> writer = new RepositoryItemWriter<>();
        writer.setRepository(customerRepo);
        writer.setMethodName("save");
        return writer;
    }

    // Step
    @Bean
    public Step step1() {
        return new StepBuilder("step-1", jobRepository)
                .<Customer, Customer>chunk(10, transactionManager)
                .reader(customerReader())
                .processor(customerProcessor())
                .writer(customerWriter())
                .build();
    }

    // Job
    @Bean
    public Job job() {
        return new JobBuilder("importCustomersJob", jobRepository)
                .start(step1())
                .build();
    }
}
