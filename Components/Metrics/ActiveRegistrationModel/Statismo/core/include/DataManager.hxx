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

#ifndef __DataManager_hxx
#define __DataManager_hxx

#include <iostream>

#include "DataManager.h"
#include "HDF5Utils.h"

namespace statismo {

////////////////////////////////////////////////
// Data manager
////////////////////////////////////////////////

template<typename T>
DataManager<T>::DataManager(const RepresenterType* representer)
    : m_representer(representer->Clone()) {
}

template<typename T>
DataManager<T>::~DataManager() {
    for (typename DataItemListType::iterator it =
                m_DataItemList.begin();
            it != m_DataItemList.end(); ++it) {
        delete (*it);
    }
    m_DataItemList.clear();
    if (m_representer) {
        m_representer->Delete();
    }

}



template<typename T>
DataManager<T>*
DataManager<T>::Load(Representer<T>* representer,
                     const std::string& filename) {
    using namespace H5;

    DataManager<T>* newDataManager = 0;

    H5File file;
    try {
        file = H5File(filename.c_str(), H5F_ACC_RDONLY);
    } catch (H5::Exception& e) {
        std::string msg(
            std::string("could not open HDF5 file \n") + e.getCDetailMsg());
        throw StatisticalModelException(msg.c_str());
    }

    try {
        // loading representer

        Group representerGroup = file.openGroup("./representer");
        std::string rep_name = HDF5Utils::readStringAttribute(representerGroup, "name");
        std::string repTypeStr = HDF5Utils::readStringAttribute(representerGroup, "datasetType");
        std::string versionStr = HDF5Utils::readStringAttribute(representerGroup, "version");
        typename RepresenterType::RepresenterDataType type = RepresenterType::TypeFromString(repTypeStr);
        if (type == RepresenterType::CUSTOM || type == RepresenterType::UNKNOWN) {
            if (rep_name != representer->GetName()) {
                std::ostringstream os;
                os << "A different representer was used to create the file and the representer is not of a standard type ";
                os << ("(RepresenterName = ") << rep_name << " does not match required name = " << representer->GetName() << ")";
                os << "Cannot load hdf5 file";
                throw StatisticalModelException(os.str().c_str());
            }
            if (versionStr != representer->GetVersion()) {
                std::ostringstream os;
                os << "The version of the representers do not match ";
                os << ("(Version = ") << versionStr << " != = " << representer->GetVersion() << ")";
                os << "Cannot load hdf5 file";

            }

        }
        if (type != representer->GetType()) {
            std::ostringstream os;
            os << "The representer that was provided cannot be used to load the dataset ";
            os << "(" << type << " != " << representer->GetType() << ").";
            os << "Cannot load hdf5 file.";
            throw StatisticalModelException(os.str().c_str());
        }

        representer->Load(representerGroup);
        representerGroup.close();
        newDataManager = new DataManager<T>(representer);


        Group publicGroup = file.openGroup("/data");
        unsigned numds = HDF5Utils::readInt(publicGroup, "./NumberOfDatasets");

        for (unsigned num = 0; num < numds; num++) {
            std::ostringstream ss;
            ss << "./dataset-" << num;

            Group dsGroup = file.openGroup(ss.str().c_str());
            newDataManager->m_DataItemList.push_back(
                DataItemType::Load(representer, dsGroup));

        }

    } catch (H5::Exception& e) {
        std::string msg(
            std::string(
                "an exception occurred while reading data matrix to HDF5 file \n")
            + e.getCDetailMsg());
        throw StatisticalModelException(msg.c_str());
    }

    file.close();

    assert(newDataManager != 0);
    return newDataManager;
}

template<typename T>
void DataManager<T>::Save(const std::string& filename) const {
    using namespace H5;

    assert(m_representer != 0);

    H5File file;

    try {
        file = H5File(filename.c_str(), H5F_ACC_TRUNC);
    } catch (H5::Exception& e) {
        std::string msg(
            std::string("Could not open HDF5 file for writing \n")
            + e.getCDetailMsg());
        throw StatisticalModelException(msg.c_str());
    }

    try {

        Group representerGroup = file.createGroup("./representer");
        std::string dataTypeStr = RepresenterType::TypeToString(m_representer->GetType());

        HDF5Utils::writeStringAttribute(representerGroup, "name", m_representer->GetName());
        HDF5Utils::writeStringAttribute(representerGroup, "version", m_representer->GetVersion());
        HDF5Utils::writeStringAttribute(representerGroup, "datasetType", dataTypeStr);

        this->m_representer->Save(representerGroup);
        representerGroup.close();


        Group publicGroup = file.createGroup("./data");
        HDF5Utils::writeInt(publicGroup, "./NumberOfDatasets",
                            this->m_DataItemList.size());

        unsigned num = 0;
        for (typename DataItemListType::const_iterator it =
                    this->m_DataItemList.begin();
                it != this->m_DataItemList.end(); ++it) {
            std::ostringstream ss;
            ss << "./dataset-" << num;

            Group dsGroup = file.createGroup(ss.str().c_str());

            (*it)->Save(dsGroup);

            dsGroup.close();
            num++;
        }
    } catch (H5::Exception& e) {
        std::string msg(
            std::string(
                "an exception occurred while writing data matrix to HDF5 file \n")
            + e.getCDetailMsg());
        throw StatisticalModelException(msg.c_str());
    }
    file.close();
}

template<typename T>
void DataManager<T>::AddDataset(DatasetConstPointerType dataset,
                                const std::string& URI) {

    DatasetPointerType sample;
    sample = m_representer->CloneDataset(dataset);

    m_DataItemList.push_back(
        DataItemType::Create(m_representer, URI,
                             m_representer->SampleToSampleVector(sample)));
    m_representer->DeleteDataset(sample);
}

template<typename T>
typename DataManager<T>::DataItemListType DataManager<T>::GetData() const {
    return m_DataItemList;
}

template<typename T>
typename DataManager<T>::CrossValidationFoldListType DataManager<T>::GetCrossValidationFolds(
    unsigned nFolds, bool randomize) const {
    if (nFolds <= 1 || nFolds > GetNumberOfSamples()) {
        throw StatisticalModelException(
            "Invalid number of folds specified in GetCrossValidationFolds");
    }
    unsigned nElemsPerFold = GetNumberOfSamples() / nFolds;

    // we create a vector with as many entries as datasets. Each entry contains the
    // fold the entry belongs to
    std::vector<unsigned> batchAssignment(GetNumberOfSamples());

    for (unsigned i = 0; i < GetNumberOfSamples(); i++) {
        batchAssignment[i] = std::min(i / nElemsPerFold, nFolds);
    }

    // randomly shuffle the vector
    srand(time(0));
    if (randomize) {
        std::random_shuffle(batchAssignment.begin(), batchAssignment.end());
    }

    // now we create the folds
    CrossValidationFoldListType foldList;
    for (unsigned currentFold = 0; currentFold < nFolds; currentFold++) {
        DataItemListType trainingData;
        DataItemListType testingData;

        unsigned sampleNum = 0;
        for (typename DataItemListType::const_iterator it =
                    m_DataItemList.begin();
                it != m_DataItemList.end(); ++it) {
            if (batchAssignment[sampleNum] != currentFold) {
                trainingData.push_back(*it);
            } else {
                testingData.push_back(*it);
            }
            ++sampleNum;
        }
        CrossValidationFoldType fold(trainingData, testingData);
        foldList.push_back(fold);
    }
    return foldList;
}

template<typename T>
typename DataManager<T>::CrossValidationFoldListType DataManager<T>::GetLeaveOneOutCrossValidationFolds() const {
    CrossValidationFoldListType foldList;
    for (unsigned currentFold = 0; currentFold < GetNumberOfSamples();
            currentFold++) {
        DataItemListType trainingData;
        DataItemListType testingData;

        unsigned sampleNum = 0;
        for (typename DataItemListType::const_iterator it =
                    m_DataItemList.begin();
                it != m_DataItemList.end(); ++it, ++sampleNum) {
            if (sampleNum == currentFold) {
                testingData.push_back(*it);
            } else {
                trainingData.push_back(*it);
            }
        }
        CrossValidationFoldType fold(trainingData, testingData);
        foldList.push_back(fold);
    }
    return foldList;
}

} // Namespace statismo

#endif
