package edu.gatech.seclass.replace;

import com.sun.javaws.exceptions.InvalidArgumentException;
import sun.java2d.pipe.NullPipe;
import sun.nio.cs.StandardCharsets;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;

class Pair {
    private String toString = "";
    private String fromString = "";

    public Pair(String to, String from)
    {
        this.toString = to;
        this.fromString = from;
    }

    public Pair()
    {

    }

    public String getTo() {
        return toString;
    }
    public void setTo(String to) {
        this.toString = to;
    }
    public String getFrom() {
        return fromString;
    }
    public void setFrom(String from) {
        this.fromString = from;
    }
    public void setToFrom(String to, String from){
        this.toString = to;
        this.fromString = from;
    }
}

class OPTS{
    public Boolean BACKUP;
    public Boolean FIRST;
    public Boolean LAST;
    public Boolean CASE;
    public Boolean NOOPTS;
    public int firstSTRindex;
    public OPTS(Boolean backup, Boolean first, Boolean last, Boolean icase, Boolean noopts, int lastIndex){
        this.BACKUP = backup;
        this.FIRST = first;
        this.LAST = last;
        this.CASE = icase;
        this.NOOPTS = noopts;
        this.firstSTRindex = lastIndex;
    }
}

public class Main {
    public static void main(String[] args) {
        // write your code here
        int separatorIndex = 0;
        String[] strings = null;
        String[] files = null;
        OPTS options = null;
        ArrayList<Pair> ToFromPairs = new ArrayList<Pair>();

        try {
            separatorIndex = Arrays.asList(args).lastIndexOf("--");

            files = Arrays.copyOfRange(args, Arrays.asList(args).lastIndexOf("--"), args.length);

            ArrayList<String[]> optionsArrays = new ArrayList<String[]>();
            String[] argsCpy = args;

            int generalIndex = 0;
            int i = 0;
            do {
                int index = Arrays.asList(argsCpy).indexOf("--");
                String[] tempArr = new String[index];
                System.arraycopy(argsCpy, 0, tempArr, 0, tempArr.length);
                optionsArrays.add(tempArr);
                generalIndex += tempArr.length+1;
                argsCpy = new String[args.length - generalIndex];
                System.arraycopy(args, generalIndex, argsCpy, 0, args.length - generalIndex);
            }while(Arrays.asList(argsCpy).contains("--"));


            options = new OPTS(false, false, false, false, false, 0);
            for (String[] s : optionsArrays){
                if(optionsArrays.size() <= 1) {
                    options = getAllOPTS(s, Arrays.asList(s).indexOf("--"), options);
                    ToFromPairs = getToFromPairs(ToFromPairs, options, Arrays.asList(s).indexOf("--"), s);
                }
                else{
                    if (optionsArrays.indexOf(s) == 0 ) {
                        options = getAllOPTS(s, Arrays.asList(s).indexOf("--"), options);

                    } else {
                        options = getAllOPTS(s, Arrays.asList(s).indexOf("--"), options);
                        ToFromPairs = getToFromPairs(ToFromPairs, options, Arrays.asList(s).indexOf("--"), s);
                    }
                }
            }

        } catch (Exception ex) {
            usage();
        }

        if (separatorIndex >= args.length) {
            usage();
        }

        //Get all content of a file and replace with to string
        if (files != null) {
            for (String oneFile : files) {
                if(oneFile != "--") {
                    try {
                        Path filePath = Paths.get(oneFile);

                        byte[] bytes = Files.readAllBytes(filePath);
                        String contents = new String(bytes);
                        String[] arr = contents.split("(?<=\n)");

                        //List<String> oldLines = Files.readAllLines(filePath);
                        List<String> oldLines = Arrays.asList(arr);
                        List<String> newLines = oldLines;

                        if (options.BACKUP == true) {
                            createBACKUP(oneFile);
                        }

                        if (options.FIRST == true) {
                            newLines = editFIRST(newLines, ToFromPairs);
                        }

                        if (options.LAST == true) {
                            newLines = editLAST(newLines, ToFromPairs);
                        }

                        if (options.CASE == true) {
                            newLines = editCASE(newLines, ToFromPairs);
                        }

                        if (options.NOOPTS == true) {
                            newLines = editNOOPS(newLines, ToFromPairs);
                        }

                        PrintWriter pw = new PrintWriter(filePath.toString());
                        for (String s : newLines) {
                            int index = newLines.indexOf(s);
//                        if (index > newLines.size() - 2) {
                            s = new StringBuilder().append(s).toString();
//                        } else {
//                            s = new StringBuilder().append(s).append('\n').toString();
//                        }
                            pw.write(s);
                        }
                        pw.close();
                    } catch (Exception ex) {
                        usage();
                    }
                }
            }
        }
    }

    private static ArrayList<Pair> getToFromPairs(ArrayList<Pair> ToFromPairs, OPTS options, int separatorIndex, String[] args) {
        String[] strings = null;

        if (options.firstSTRindex < 0){
            ToFromPairs.add(new Pair("", ""));
        }
        else{
            strings = Arrays.copyOfRange(args, options.firstSTRindex, args.length);
        }

        if (strings != null){
            for (int i = 0; i < strings.length; i++) {
                Pair pair = new Pair();
                if (i < strings.length) {
                    pair.setFrom(strings[i]);
                    i++;
                } else {
                    pair.setFrom("");
                }
                if (i < strings.length) {
                    pair.setTo(strings[i]);
                    i++;
                } else {
                    pair.setTo("");
                }
                ToFromPairs.add(pair);
            }
        }
        System.out.println("End method");

        return ToFromPairs;
    }

    private static void createBACKUP (String oneFile){
        Path filePath = Paths.get(oneFile);
        Path source = Paths.get(filePath.toString());
        Path target = Paths.get(filePath.toString() + ".bck");
        try {
            Files.copy(source, target);
        } catch (IOException e1) {
            e1.printStackTrace();
        }
    }

    private static List<String> editFIRST(List<String> newLines, ArrayList<Pair> ToFromPairs){
        String newContent;
        int index = 0;
        String oneLine = newLines.get(index);
        for (Pair pair : ToFromPairs) {
            if (oneLine.contains(pair.getFrom())) {
                int iStart = oneLine.indexOf(pair.getFrom());
                oneLine = oneLine.replace(oneLine.substring(iStart, iStart + pair.getFrom().length()), pair.getTo());
            }
        }
        newContent = new StringBuilder().append(oneLine).toString();
        newLines.set(index, newContent);
        return newLines;
    }

    private static List<String> editLAST(List<String> newLines, ArrayList<Pair> ToFromPairs){
        String newContent;
        int index = newLines.size()-1;
        String oneLine = newLines.get(index);
        for (Pair pair : ToFromPairs) {
            if (oneLine.contains(pair.getFrom())) {
                int iStart = oneLine.indexOf(pair.getFrom());
                oneLine = oneLine.replace(oneLine.substring(iStart, iStart + pair.getFrom().length()), pair.getTo());
            }
        }
        newContent = new StringBuilder().append(oneLine).toString();
        newLines.set(index, newContent);
        return newLines;
    }

    private static List<String> editCASE (List<String> newLines, ArrayList<Pair> ToFromPairs){

        System.out.println("Enter editCASE");
        String newContent;
        for (String oneLine : newLines)
        {
            System.out.println("Enter 1st for loop");
            int index = newLines.indexOf(oneLine);
            for (Pair pair : ToFromPairs) {
                System.out.println("Enter 2nd fr loop");
                String lowFromString = pair.getFrom().toLowerCase();
                if (oneLine.toLowerCase().contains(lowFromString)) {
                    System.out.println("Enter if statement");
                    int iStart = oneLine.toLowerCase().indexOf(lowFromString);
                    oneLine = oneLine.replace(oneLine.substring(iStart, iStart + lowFromString.length()), pair.getTo());
                }
            }

            newContent = new StringBuilder().append(oneLine).toString();
            newLines.set(index, newContent);
        }
        return newLines;
    }

    private static List<String> editNOOPS (List<String> newLines, ArrayList<Pair> ToFromPairs){
        String newContent;
        for (String oneLine : newLines) {
            int index = newLines.indexOf(oneLine);
            for (Pair pair : ToFromPairs) {
                if (oneLine.contains(pair.getFrom())) {
                    int iStart = oneLine.indexOf(pair.getFrom());
                    oneLine = oneLine.replace(oneLine.substring(iStart, iStart + pair.getFrom().length()), pair.getTo());
                }
            }

            newContent = new StringBuilder().append(oneLine).toString();
            newLines.set(index, newContent);
        }
        return newLines;

    }

    private static OPTS getAllOPTS(String[] args, int separatorIndex, OPTS options){
        options.firstSTRindex = -1;
//        if(separatorIndex<2) {
//            System.out.println("Enter separatorIndex" + separatorIndex);
//            options.NOOPTS = true;
//            return options;
//        }
//        else {
        if(args.length>0){
            System.out.println("Enter else");

            options.NOOPTS = false;
            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "-b":
                        options.BACKUP = true;
                        break;//create backup copy of each file on which replace operation is performed before modification,
                    case "-f":
                        options.FIRST = true;
                        break;//only replace the first occurence of string in each file
                    case "-l":
                        options.LAST = true;
                        break;//only replace the last occurence of string in each file
                    case "-i":
                        options.CASE = true;
                        break;//search for strings in a case insensitive way.
                    default:
                        if(options.firstSTRindex<0){options.firstSTRindex = i;}
                        break;
                        //i = separatorIndex;
                }
            }
        }

        if(args.length == 0 || (options.BACKUP == false && options.CASE == false && options.FIRST ==false && options.LAST == false)){
            options.NOOPTS = true;
        }

        return options;
    }

    private static void usage() {
        System.err.println("Usage: Replace [-b] [-f] [-l] [-i] <from> <to> -- " + "<filename> [<filename>]*" );
    }
}