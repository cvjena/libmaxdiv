//  Copyright (C) 2016 Bjoern Barz (University of Jena)
//
//  This file is part of libmaxdiv.
//
//  libmaxdiv is free software: you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  libmaxdiv is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
//
//  You should have received a copy of the GNU Lesser General Public License
//  along with libmaxdiv. If not, see <http://www.gnu.org/licenses/>.

#include "utils.h"
#include <cctype>
#include <fstream>
using namespace MaxDiv;
using namespace std;


string MaxDiv::trim(string str)
{
    size_t pos = str.find_first_not_of(" \r\n\t");
    if (pos == string::npos)
        return "";
    else if (pos > 0)
        str = str.substr(pos);
    pos = str.find_last_not_of(" \r\n\t");
    if (pos < str.length() - 1 && pos != string::npos)
        str = str.substr(0, pos + 1);
    return str;
}

string MaxDiv::strtolower(const string & str)
{
    string lstr(str.length(), '\0');
    string::const_iterator itIn;
    string::iterator itOut;
    for (itIn = str.begin(), itOut = lstr.begin(); itIn != str.end(); itIn++, itOut++)
        *itOut = tolower(*itIn);
    return lstr;
}

string MaxDiv::strtoupper(const string & str)
{
    string lstr(str.length(), '\0');
    string::const_iterator itIn;
    string::iterator itOut;
    for (itIn = str.begin(), itOut = lstr.begin(); itIn != str.end(); itIn++, itOut++)
        *itOut = toupper(*itIn);
    return lstr;
}

int MaxDiv::splitString(const string & str, const char * delimiters, vector<string> & tokens)
{
    int numTokens = 0;
    size_t pos = 0, startPos = 0;
    do
    {
        startPos = str.find_first_not_of(delimiters, pos);
        pos = str.find_first_of(delimiters, startPos);
        if (startPos < pos)
        {
            tokens.push_back(str.substr(startPos, pos - startPos));
            numTokens++;
        }
    }
    while (pos != string::npos);
    return numTokens;
}

DataTensor MaxDiv::readDataFromCSV(const string & filename, char delimiter, DataTensor::Index firstRow, DataTensor::Index firstCol, DataTensor::Index lastCol)
{
    ifstream file(filename.c_str());
    if (!file.is_open())
        return DataTensor();
    
    // 1st pass: determine number of rows and columns in the file
    DataTensor::Index rows = 0, cols = 1;
    string line;
    while (!file.eof())
    {
        getline(file, line);
        if (rows == firstRow)
        {
            for (string::const_iterator c = line.begin(); c != line.end(); ++c)
                if (*c == delimiter)
                    ++cols;
        }
        if (trim(line) != "")
            ++rows;
    }
    if (firstRow >= rows || firstCol >= cols)
        return DataTensor();
    
    // 2nd pass: read data
    DataTensor data({ rows - firstRow, 1, 1, 1, ((lastCol >= cols) ? cols : lastCol + 1) - firstCol });
    vector<string> fields;
    char * str_end;
    char delimiters[2];
    delimiters[0] = delimiter;
    delimiters[1] = '\0';
    DataTensor::Index row = 0, col;
    file.clear();
    file.seekg(0);
    while (!file.eof())
    {
        getline(file, line);
        line = trim(line);
        if (line != "" && row >= firstRow)
        {
            if (static_cast<DataTensor::Index>(splitString(line, delimiters, fields)) != cols)
                return DataTensor();
            for (col = firstCol; col < cols && col <= lastCol; ++col)
            {
                data({ row - firstRow, 0, 0, 0, col - firstCol }) = strtod(fields[col].c_str(), &str_end);
                if (str_end == fields[col].c_str())
                    return DataTensor();
            }
            fields.clear();
        }
        if (line != "")
            ++row;
    }
    return data;
}
