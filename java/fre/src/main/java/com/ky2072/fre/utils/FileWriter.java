// edited by ky2072 KeYang 2022-05-17
package com.ky2072.fre.utils;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Map;
import java.util.Objects;

@Component
public class FileWriter {
    /**
     * Class for writing data into disks
     */

    @Value("${fre.path.data-path}")
    private String dataPath;

    public void writeToCSV(ArrayList<Map<String, String>> list) {
        /*
          write a list of map into csv
         */

        try {
            BufferedWriter in = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(dataPath + "/input.csv"), "UTF-8"));
            BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(dataPath + "/output.csv"), "UTF-8"));

            if (list == null) return;

            for (Map.Entry<String, String> entry : list.get(0).entrySet()) {
                out.write(entry.getKey());
                out.write(",");

                if (Objects.equals(entry.getKey(), "optionPrice")) continue;
                in.write(entry.getKey());
                in.write(",");
            }
            out.newLine();
            in.newLine();

            for (Map<String, String> map : list) {
                for (Map.Entry<String, String> entry : map.entrySet()) {
                    out.write(entry.getValue());
                    out.write(",");

                    if (Objects.equals(entry.getKey(), "optionPrice")) continue;
                    in.write(entry.getValue());
                    in.write(",");
                }
                out.newLine();
                in.newLine();
            }

            in.flush();
            in.close();
            out.flush();
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
