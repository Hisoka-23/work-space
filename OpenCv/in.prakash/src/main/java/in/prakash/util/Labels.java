package in.prakash.util;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class Labels {
    private final File file;
    private Map<Integer,String> idToName = new HashMap<>();

    public Labels(File file) {
        this.file = file;
        if (file.exists()) {
            try {
                Map<String,String> m = new ObjectMapper().readValue(file, Map.class);
                idToName = new HashMap<>();
                for (Map.Entry<String,String> e : m.entrySet()) {
                    idToName.put(Integer.parseInt(e.getKey()), e.getValue());
                }
            } catch (Exception ignored) { }
        }
    }

    public void saveMap(Map<Integer,String> map) {
        this.idToName = new HashMap<>(map);
        Map<String,String> s = new LinkedHashMap<>();
        for (Map.Entry<Integer,String> e : idToName.entrySet()) s.put(Integer.toString(e.getKey()), e.getValue());
        file.getParentFile().mkdirs();
        try { new ObjectMapper().writeValue(file, s); }
        catch (IOException e) { throw new RuntimeException(e); }
    }

    public String nameFor(int id) {
        return idToName.getOrDefault(id, "ID " + id);
    }
}
