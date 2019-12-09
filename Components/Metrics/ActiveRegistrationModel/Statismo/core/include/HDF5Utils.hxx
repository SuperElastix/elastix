/*
 * This file is part of the statismo library.
 *
 * Author: Marcel Luethi (marcel.luethi@unibas.ch)
 *
 * Copyright (c) 2011 University of Basel
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * Neither the name of the project's author nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef __HDF5_UTILS_CXX
#define __HDF5_UTILS_CXX

#include "HDF5Utils.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>

#include "CommonTypes.h"
#include "Exceptions.h"
#include "itk_H5Cpp.h"

namespace statismo {

inline
H5::H5File
HDF5Utils::openOrCreateFile(const std::string filename) {

    // check if file exists
    std::ifstream ifile(filename.c_str());
    H5::H5File file;

    if (!ifile) {
        // create it
        file = H5::H5File( filename.c_str(), H5F_ACC_EXCL);
    } else {
        // open it
        file = H5::H5File( filename.c_str(), H5F_ACC_RDWR);
    }
    return file;
}



inline
H5::Group
HDF5Utils::openPath(H5::H5File& file, const std::string& path, bool createPath) {
    H5::Group group;

    // take the first part of the path
    size_t curpos = 1;
    size_t nextpos = path.find_first_of("/", curpos);
    H5::Group g = file.openGroup("/");

    std::string name = path.substr(curpos, nextpos-1);

    while (curpos != std::string::npos && name != "") {

        if (existsObjectWithName(g, name)) {
            g = g.openGroup(name);
        } else {
            if (createPath) {
                g = g.createGroup(name);
            } else {
                std::string msg = std::string("the path ") +path +" does not exist";
                throw StatisticalModelException(msg.c_str());
            }
        }

        curpos = nextpos+1;
        nextpos = path.find_first_of("/", curpos);
        if ( nextpos != std::string::npos )
            name = path.substr(curpos, nextpos-curpos);
        else
            name = path.substr(curpos);
    }

    return g;
}

template <class T>
inline
void HDF5Utils::readMatrixOfType(const H5::H5Location& fg, const char* name, typename GenericEigenType<T>::MatrixType& matrix) {
    throw StatisticalModelException("Invalid type proided for writeMatrixOfType");
}

template <>
inline
void HDF5Utils::readMatrixOfType<unsigned int>(const H5::H5Location& fg, const char* name, GenericEigenType<unsigned int>::MatrixType& matrix) {
    H5::DataSet ds = fg.openDataSet( name );
    hsize_t dims[2];
    ds.getSpace().getSimpleExtentDims(dims, NULL);

    // simply read the whole dataspace
    matrix.resize(dims[0], dims[1]);
    ds.read(matrix.data(), H5::PredType::NATIVE_UINT);
}

template <>
inline
void HDF5Utils::readMatrixOfType<float>(const H5::H5Location& fg, const char* name, GenericEigenType<float>::MatrixType& matrix) {
    H5::DataSet ds = fg.openDataSet( name );
    hsize_t dims[2];
    ds.getSpace().getSimpleExtentDims(dims, NULL);

    // simply read the whole dataspace
    matrix.resize(dims[0], dims[1]);
    ds.read(matrix.data(), H5::PredType::NATIVE_FLOAT);
}

template <>
inline
void HDF5Utils::readMatrixOfType<double>(const H5::H5Location& fg, const char* name, GenericEigenType<double>::MatrixType& matrix) {
    H5::DataSet ds = fg.openDataSet( name );
    hsize_t dims[2];
    ds.getSpace().getSimpleExtentDims(dims, NULL);

    // simply read the whole dataspace
    matrix.resize(dims[0], dims[1]);
    ds.read(matrix.data(), H5::PredType::NATIVE_DOUBLE);
}


inline
void HDF5Utils::readMatrix(const H5::H5Location& fg, const char* name, MatrixType& matrix) {
    readMatrixOfType<ScalarType>(fg, name, matrix);
}


inline
void HDF5Utils::readMatrix(const H5::H5Location& fg, const char* name, unsigned maxNumColumns, MatrixType& matrix) {
    H5::DataSet ds = fg.openDataSet( name );
    hsize_t dims[2];
    ds.getSpace().getSimpleExtentDims(dims, NULL);

    hsize_t nRows = dims[0]; // take the number of rows defined in the hdf5 file
    hsize_t nCols = std::min(dims[1], static_cast<hsize_t>(maxNumColumns)); // take the number of cols provided by the user

    hsize_t offset[2] = {0,0};   // hyperslab offset in the file
    hsize_t count[2];
    count[0] = nRows;
    count[1] =  nCols;

    H5::DataSpace dataspace = ds.getSpace();
    dataspace.selectHyperslab( H5S_SELECT_SET, count, offset );

    /* Define the memory dataspace. */
    hsize_t     dimsm[2];
    dimsm[0] = nRows;
    dimsm[1] = nCols;
    H5::DataSpace memspace( 2, dimsm );

    /* Define memory hyperslab. */
    hsize_t      offset_out[2] = {0, 0};       // hyperslab offset in memory
    hsize_t      count_out[2];        // size of the hyperslab in memory

    count_out[0]  = nRows;
    count_out[1] = nCols;
    memspace.selectHyperslab( H5S_SELECT_SET, count_out, offset_out );

    matrix.resize(nRows, nCols);
    // ds.read(matrix.data(), H5::PredType::NATIVE_FLOAT, memspace, dataspace);
    ds.read(matrix.data(), H5::PredType::NATIVE_DOUBLE, memspace, dataspace);

}

template <class T>
inline
H5::DataSet HDF5Utils::writeMatrixOfType(const H5::H5Location& fg, const char* name, const typename GenericEigenType<T>::MatrixType& matrix) {
    throw StatisticalModelException("Invalid type proided for writeMatrixOfType");
}

template <>
inline
H5::DataSet HDF5Utils::writeMatrixOfType<unsigned int>(const H5::H5Location& fg, const char* name, const GenericEigenType<unsigned int>::MatrixType& matrix) {
    // HDF5 does not like empty matrices.
    //
    if (matrix.rows() == 0 || matrix.cols() == 0) {
        throw StatisticalModelException("Empty matrix provided to writeMatrix");
    }

    hsize_t dims[2] = {static_cast<hsize_t>(matrix.rows()), static_cast<hsize_t>(matrix.cols())};
    H5::DataSet ds = fg.createDataSet( name, H5::PredType::NATIVE_UINT, H5::DataSpace(2, dims));
    ds.write( matrix.data(), H5::PredType::NATIVE_UINT );
    return ds;
}

template <>
inline
H5::DataSet HDF5Utils::writeMatrixOfType<float>(const H5::H5Location& fg, const char* name, const GenericEigenType<float>::MatrixType& matrix) {
    // HDF5 does not like empty matrices.
    //
    if (matrix.rows() == 0 || matrix.cols() == 0) {
        throw StatisticalModelException("Empty matrix provided to writeMatrix");
    }

    hsize_t dims[2] = {static_cast<hsize_t>(matrix.rows()), static_cast<hsize_t>(matrix.cols())};
    H5::DataSet ds = fg.createDataSet( name, H5::PredType::NATIVE_FLOAT, H5::DataSpace(2, dims));
    ds.write( matrix.data(), H5::PredType::NATIVE_FLOAT );
    return ds;
}

template <>
inline
H5::DataSet HDF5Utils::writeMatrixOfType<double>(const H5::H5Location& fg, const char* name, const GenericEigenType<double>::MatrixType& matrix) {
    // HDF5 does not like empty matrices.
    //
    if (matrix.rows() == 0 || matrix.cols() == 0) {
        throw StatisticalModelException("Empty matrix provided to writeMatrix");
    }

    hsize_t dims[2] = {static_cast<hsize_t>(matrix.rows()), static_cast<hsize_t>(matrix.cols())};
    H5::DataSet ds = fg.createDataSet( name, H5::PredType::NATIVE_DOUBLE, H5::DataSpace(2, dims));
    ds.write( matrix.data(), H5::PredType::NATIVE_DOUBLE );
    return ds;
}


inline
H5::DataSet HDF5Utils::writeMatrix(const H5::H5Location& fg, const char* name, const MatrixType& matrix) {
    return writeMatrixOfType<ScalarType>(fg, name, matrix);
}


template <class T>
inline
void HDF5Utils::readVectorOfType(const H5::H5Location& fg, const char* name, typename GenericEigenType<T>::VectorType& vector) {
    throw StatisticalModelException("Invalid type proided for readVectorOfType");
}

template <>
inline
void HDF5Utils::readVectorOfType<double>(const H5::H5Location& fg, const char* name,  GenericEigenType<double>::VectorType& vector) {
    H5::DataSet ds = fg.openDataSet( name );
    hsize_t dims[1];
    ds.getSpace().getSimpleExtentDims(dims, NULL);
    vector.resize(dims[0], 1);
    ds.read(vector.data(), H5::PredType::NATIVE_DOUBLE);
}

template <>
inline
void HDF5Utils::readVectorOfType<float>(const H5::H5Location& fg, const char* name,  GenericEigenType<float>::VectorType& vector) {
    H5::DataSet ds = fg.openDataSet( name );
    hsize_t dims[1];
    ds.getSpace().getSimpleExtentDims(dims, NULL);
    vector.resize(dims[0], 1);
    ds.read(vector.data(), H5::PredType::NATIVE_FLOAT);
}

template <>
inline
void HDF5Utils::readVectorOfType<int>(const H5::H5Location& fg, const char* name,  GenericEigenType<int>::VectorType& vector) {
    H5::DataSet ds = fg.openDataSet( name );
    hsize_t dims[1];
    ds.getSpace().getSimpleExtentDims(dims, NULL);
    vector.resize(dims[0], 1);
    ds.read(vector.data(), H5::PredType::NATIVE_INT);
}

inline
void HDF5Utils::readVector(const H5::H5Location& fg, const char* name, VectorType& vector) {
    readVectorOfType<ScalarType>(fg, name, vector);
}


inline
void HDF5Utils::readVector(const H5::H5Location& fg, const char* name, unsigned maxNumElements, VectorType& vector) {
    H5::DataSet ds = fg.openDataSet( name );
    hsize_t dims[1];
    ds.getSpace().getSimpleExtentDims(dims, NULL);

    hsize_t nElements = std::min(dims[0], static_cast<hsize_t>(maxNumElements)); // take the number of rows defined in the hdf5 file

    hsize_t offset[1] = {0};   // hyperslab offset in the file
    hsize_t count[1];
    count[0] = nElements;

    H5::DataSpace dataspace = ds.getSpace();
    dataspace.selectHyperslab( H5S_SELECT_SET, count, offset );

    /* Define the memory dataspace. */
    hsize_t     dimsm[1];
    dimsm[0] = nElements;
    H5::DataSpace memspace( 1, dimsm );

    /* Define memory hyperslab. */
    hsize_t      offset_out[1] = {0};       // hyperslab offset in memory
    hsize_t      count_out[1];        // size of the hyperslab in memory

    count_out[0]  = nElements;
    memspace.selectHyperslab( H5S_SELECT_SET, count_out, offset_out );

    vector.resize(nElements);
    HDF5Utils::readVector(fg, name, vector);
}




template <class T>
inline
H5::DataSet HDF5Utils::writeVectorOfType(const H5::H5Location& fg, const char* name, const typename GenericEigenType<T>::VectorType& vector) {
    throw StatisticalModelException("Invalid type provided for writeVectorOfType");
}

template <>
inline
H5::DataSet HDF5Utils::writeVectorOfType<double>(const H5::H5Location& fg, const char* name, const GenericEigenType<double>::VectorType& vector) {
    hsize_t dims[1] = {static_cast<hsize_t>(vector.size())};
    H5::DataSet ds = fg.createDataSet( name, H5::PredType::NATIVE_DOUBLE, H5::DataSpace(1, dims));
    ds.write( vector.data(), H5::PredType::NATIVE_DOUBLE );
    return ds;
}

template <>
inline
H5::DataSet HDF5Utils::writeVectorOfType<float>(const H5::H5Location& fg, const char* name, const GenericEigenType<float>::VectorType& vector) {
    hsize_t dims[1] = {static_cast<hsize_t>(vector.size())};
    H5::DataSet ds = fg.createDataSet( name, H5::PredType::NATIVE_FLOAT, H5::DataSpace(1, dims));
    ds.write( vector.data(), H5::PredType::NATIVE_FLOAT );
    return ds;
}

template <>
inline
H5::DataSet HDF5Utils::writeVectorOfType<int>(const H5::H5Location& fg, const char* name, const GenericEigenType<int>::VectorType& vector) {
    hsize_t dims[1] = {static_cast<hsize_t>(vector.size())};
    H5::DataSet ds = fg.createDataSet( name, H5::PredType::NATIVE_INT, H5::DataSpace(1, dims));
    ds.write( vector.data(), H5::PredType::NATIVE_INT );
    return ds;
}

inline
H5::DataSet HDF5Utils::writeVector(const H5::H5Location& fg, const char* name, const VectorType& vector) {
    return writeVectorOfType<ScalarType>(fg, name, vector);
}


inline
H5::DataSet HDF5Utils::writeString(const H5::H5Location& fg, const char* name, const std::string& s) {
    H5::StrType fls_type(H5::PredType::C_S1, s.length() + 1); // + 1 for trailing zero
    H5::DataSet ds = fg.createDataSet(name, fls_type, H5::DataSpace(H5S_SCALAR));
    ds.write(s, fls_type);
    return ds;
}


inline
std::string
HDF5Utils::readString(const H5::H5Location& fg, const char* name) {
    H5std_string outputString;
    H5::DataSet ds = fg.openDataSet(name);
    ds.read(outputString, ds.getStrType());
    return outputString;
}

inline
void HDF5Utils::writeStringAttribute(const H5::H5Object& fg, const char* name, const std::string& s) {
    H5::StrType strdatatype(H5::PredType::C_S1, s.length() + 1 ); // + 1 for trailing 0
    H5::Attribute att = fg.createAttribute(name, strdatatype, H5::DataSpace(H5S_SCALAR));
    att.write(strdatatype, s);
    att.close();
}


inline
std::string
HDF5Utils::readStringAttribute(const H5::H5Object& fg, const char* name) {
    H5std_string outputString;

    H5::Attribute myatt_out = fg.openAttribute(name);
    myatt_out.read(myatt_out.getStrType(), outputString);
    return outputString;
}

inline
void HDF5Utils::writeIntAttribute(const H5::H5Object& fg, const char* name, int value) {
    H5::IntType int_type(H5::PredType::NATIVE_INT32);
    H5::DataSpace att_space(H5S_SCALAR);
    H5::Attribute att = fg.createAttribute(name, int_type, att_space );
    att.write( int_type, &value);
    att.close();
}

inline
int
HDF5Utils::readIntAttribute(const H5::H5Object& fg, const char* name) {
    H5::IntType fls_type(H5::PredType::NATIVE_INT32);
    int value = 0;
    H5::Attribute myatt_out = fg.openAttribute(name);
    myatt_out.read(fls_type, &value);
    return value;
}


inline
H5::DataSet HDF5Utils::writeInt(const H5::H5Location& fg, const char* name, int value) {
    H5::IntType fls_type(H5::PredType::NATIVE_INT32); // 0 is a dummy argument
    H5::DataSet ds = fg.createDataSet(name, fls_type, H5::DataSpace(H5S_SCALAR));
    ds.write(&value, fls_type);
    return ds;
}

inline
int HDF5Utils::readInt(const H5::H5Location& fg, const char* name) {
    H5::IntType fls_type(H5::PredType::NATIVE_INT32);
    H5::DataSet ds = fg.openDataSet( name );

    int value = 0;
    ds.read(&value, fls_type);
    return value;
}

inline
H5::DataSet HDF5Utils::writeFloat(const H5::H5Location& fg, const char* name, float value) {
    H5::FloatType fls_type(H5::PredType::NATIVE_FLOAT); // 0 is a dummy argument
    H5::DataSet ds = fg.createDataSet(name, fls_type, H5::DataSpace(H5S_SCALAR));
    ds.write(&value, fls_type);
    return ds;
}

inline
float HDF5Utils::readFloat(const H5::H5Location& fg, const char* name) {
    H5::FloatType fls_type(H5::PredType::NATIVE_FLOAT);
    H5::DataSet ds = fg.openDataSet( name );

    float value = 0;
    ds.read(&value, fls_type);
    return value;
}

inline
void HDF5Utils::getFileFromHDF5(const H5::H5Location& fg, const char* name, const char* filename) {
    H5::DataSet ds = fg.openDataSet( name );
    hsize_t dims[1];
    ds.getSpace().getSimpleExtentDims(dims, NULL);
    std::vector<char> buffer(dims[0]);
    if(!buffer.empty()) ds.read(&buffer[0], H5::PredType::NATIVE_CHAR);

    typedef std::ostream_iterator<char> ostream_iterator;
    std::ofstream ofile(filename, std::ios::binary);
    if (!ofile) {
        std::string s= std::string("could not open file ") +filename;
        throw StatisticalModelException(s.c_str());
    }

    std::copy(buffer.begin(), buffer.end(), ostream_iterator(ofile));
    ofile.close();
}

inline
void
HDF5Utils::dumpFileToHDF5( const char* filename, const H5::H5Location& fg, const char* name) {

    typedef std::istream_iterator<char> istream_iterator;

    std::ifstream ifile(filename, std::ios::binary);
    if (!ifile) {
        std::string s= std::string("could not open file ") +filename;
        throw StatisticalModelException(s.c_str());
    }

    std::vector<char> buffer;
    ifile >> std::noskipws;
    std::copy(istream_iterator(ifile), istream_iterator(), std::back_inserter(buffer));

    ifile.close();

    hsize_t dims[] = {buffer.size()};
    H5::DataSet ds = fg.createDataSet( name,  H5::PredType::NATIVE_CHAR, H5::DataSpace(1, dims));
    ds.write( &buffer[0], H5::PredType::NATIVE_CHAR );

}

template<typename T>
inline
void
HDF5Utils::readArray(const H5::H5Location& fg, const char* name, std::vector<T> & array) {
    throw StatisticalModelException( "not implemented" );
}


template<typename T>
inline
H5::DataSet
HDF5Utils::writeArray(const H5::H5Location& fg, const char* name, std::vector<T> const& array) {
    throw StatisticalModelException( "not implemented" );
}

template<>
inline
void
HDF5Utils::readArray(const H5::H5Location& fg, const char* name, std::vector<int> & array) {
    H5::DataSet ds = fg.openDataSet( name );
    hsize_t dims[1];
    ds.getSpace().getSimpleExtentDims(dims, NULL);
    array.resize(dims[0]);
    ds.read( &array[0], H5::PredType::NATIVE_INT32);
}

template<>
inline
H5::DataSet
HDF5Utils::writeArray(const H5::H5Location& fg, const char* name, std::vector<int> const& array) {
    hsize_t dims[1] = {array.size()};
    H5::DataSet ds = fg.createDataSet( name, H5::PredType::NATIVE_INT32, H5::DataSpace(1, dims));
    ds.write( &array[0], H5::PredType::NATIVE_INT32 );
    return ds;
}

inline
bool
HDF5Utils::existsObjectWithName(const H5::H5Location& fg, const std::string& name) {
    for (hsize_t i = 0; i < fg.getNumObjs(); ++i) {
        std::string objname= 	fg.getObjnameByIdx(i);
        if (objname == name) {
            return true;
        }
    }
    return false;
}

} //namespace statismo

#endif

