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

#include <ctime>
#include <iostream>

#include "DataManager.h"
#include "Exceptions.h"
#include "HDF5Utils.h"
#include "ModelInfo.h"

namespace statismo {

ModelInfo::ModelInfo() {
}

ModelInfo::ModelInfo(const MatrixType &scores, const ModelInfo::BuilderInfoList &builderInfos)
    : m_scores(scores), m_builderInfo(builderInfos) {
}

ModelInfo::ModelInfo(const MatrixType &scores)
    : m_scores(scores) {
}

ModelInfo::~ModelInfo() {}

ModelInfo &ModelInfo::operator=(const ModelInfo &rhs) {
    if (this == &rhs) {
        return *this;
    }
    this->m_builderInfo = rhs.m_builderInfo;
    this->m_scores = rhs.m_scores;
    return *this;
}

ModelInfo::BuilderInfoList ModelInfo::GetBuilderInfoList() const {
    return m_builderInfo;
}

const MatrixType &ModelInfo::GetScoresMatrix() const {
    return m_scores;
}

void
ModelInfo::Save(const H5::CommonFG& publicFg) const {
    using namespace H5;

    // get time and date
    time_t rawtime;
    struct tm * timeinfo;
    std::time ( &rawtime );
    timeinfo = std::localtime ( &rawtime );


    try {
        Group publicInfo = publicFg.createGroup("./modelinfo");
        HDF5Utils::writeString(publicInfo, "./build-time", std::asctime (timeinfo));
        if (m_scores.rows() != 0 && m_scores.cols() != 0) {
            HDF5Utils::writeMatrix(publicInfo, "./scores", m_scores);
        } else {
            // HDF5 does not allow us to write empty matrices. Therefore, we write a dummy matrix with 1 element
            HDF5Utils::writeMatrix(publicInfo, "./scores", MatrixType::Zero(1,1));
        }


        for (unsigned i =0; i < m_builderInfo.size(); i++) {
            std::ostringstream ss;
            ss << "./modelBuilder-" << i;
            Group modelBuilderGroup = publicInfo.createGroup(ss.str().c_str());
            m_builderInfo[i].Save(modelBuilderGroup);
            modelBuilderGroup.close();
        }

        publicInfo.close();

    } catch (H5::Exception& e) {
        std::string msg(std::string("an exception occurred while writing model info HDF5 file \n") + e.getCDetailMsg());
        throw StatisticalModelException(msg.c_str());
    }

}

void
ModelInfo::Load(const H5::CommonFG& publicFg) {
    using namespace H5;
    Group publicModelGroup = publicFg.openGroup("./modelinfo");
    try {
        HDF5Utils::readMatrix(publicModelGroup, "./scores", m_scores);
    } catch (H5::Exception& e) {
        // the likely cause is that there are no scores. so we set them as empty
        m_scores.resize(0,0);
    }

    if (m_scores.cols() == 1 &&  m_scores.rows() == 1 && m_scores(0,0) == 0.0) {
        // we observed a dummy matrix, that was created when saving the model info.
        // This means that no scores have been saved.
        m_scores.resize(0,0);
    }

    m_builderInfo.clear();
    unsigned numEntries = publicModelGroup.getNumObjs();

    for (unsigned i = 0; i < numEntries; i++) {
        H5std_string key = publicModelGroup.getObjnameByIdx(i);

        // Compatibility to older statismo file-format.
        // if we find at this level a dataInfo object, then it needs to be an old statismo file.
        if (key.find("dataInfo") != std::string::npos || key.find("builderInfo") != std::string::npos) {
            BuilderInfo bi = LoadDataInfoOldStatismoFormat(publicModelGroup);
            m_builderInfo.push_back(bi);
            // we have all the information that is stored in the info block of an old statismo file.
            // hence we can leave
            break;

        }

        // check for all modelBuilder objects and compile them into a list
        if (key.find("modelBuilder") != std::string::npos) {

            Group modelBuilderGroup = publicModelGroup.openGroup(key.c_str());
            BuilderInfo bi;
            bi.Load(modelBuilderGroup);
            m_builderInfo.push_back(bi);
        }

    }
    publicModelGroup.close();
}

inline
BuilderInfo
ModelInfo::LoadDataInfoOldStatismoFormat(const H5::CommonFG& publicModelGroup) const {
    using namespace H5;

    Group dataInfoGroup = publicModelGroup.openGroup("./dataInfo");
    BuilderInfo::KeyValueList dataInfo;
    BuilderInfo::FillKeyValueListFromInfoGroup(dataInfoGroup, dataInfo);
    dataInfoGroup.close();

    Group builderInfoGroup = publicModelGroup.openGroup("./builderInfo");
    BuilderInfo::KeyValueList paramInfo;
    BuilderInfo::FillKeyValueListFromInfoGroup(builderInfoGroup, paramInfo);

    std::string buildTime= HDF5Utils::readString(publicModelGroup,"build-time");

    // add the information to a new BuilderInfo object
    // as a first step we need to find the builderName from the parameter list
    std::string builderName = "";
    for (BuilderInfo::KeyValueList::iterator it = paramInfo.begin(); it != paramInfo.end(); it++) {
        if (it->first.find("BuilderName") != std::string::npos) {
            builderName = it->second;
            paramInfo.erase(it);
            break;
        }
    }

    return BuilderInfo(builderName, buildTime, dataInfo, paramInfo);

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// BuilderInfo
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

BuilderInfo::BuilderInfo(const std::string &modelBuilderName, const std::string &buildTime, const BuilderInfo::DataInfoList &di, const BuilderInfo::ParameterInfoList &pi)
    : m_modelBuilderName(modelBuilderName), m_buildtime(buildTime), m_dataInfo(di), m_parameterInfo(pi) {
}

BuilderInfo::BuilderInfo(const std::string &modelBuilderName, const BuilderInfo::DataInfoList &di, const BuilderInfo::ParameterInfoList &pi)
    : m_modelBuilderName(modelBuilderName), m_dataInfo(di), m_parameterInfo(pi) {

    // get time and date
    time_t rawtime;
    struct tm * timeinfo;

    std::time ( &rawtime );
    timeinfo = std::localtime ( &rawtime );
    m_buildtime = std::asctime (timeinfo);

}

BuilderInfo::BuilderInfo() {}

BuilderInfo::~BuilderInfo() {}

BuilderInfo &BuilderInfo::operator=(const BuilderInfo &rhs) {
    if (this == &rhs) {
        return *this;
    }
    this->m_modelBuilderName =rhs.m_modelBuilderName;
    this->m_buildtime = rhs.m_buildtime;
    this->m_dataInfo = rhs.m_dataInfo;
    this->m_parameterInfo = rhs.m_parameterInfo;
    return *this;
}

BuilderInfo::BuilderInfo(const BuilderInfo &orig) {
    operator=(orig);
}

void
BuilderInfo::Save(const H5::CommonFG& modelBuilderGroup) const {
    using namespace H5;

    try {
        HDF5Utils::writeString(modelBuilderGroup, "./builderName", m_modelBuilderName);
        HDF5Utils::writeString(modelBuilderGroup, "./buildTime", m_buildtime);

        Group dataInfoGroup = modelBuilderGroup.createGroup("./dataInfo");
        for (DataInfoList::const_iterator it = m_dataInfo.begin(); it != m_dataInfo.end(); ++it) {
            HDF5Utils::writeString(dataInfoGroup, it->first.c_str(), it->second.c_str());
        }


        dataInfoGroup.close();

        Group parameterGroup = modelBuilderGroup.createGroup("./parameters");
        for (ParameterInfoList::const_iterator it = m_parameterInfo.begin();	it != m_parameterInfo.end(); ++it) {
            HDF5Utils::writeString(parameterGroup, it->first.c_str(), it->second.c_str());
        }

        parameterGroup.close();

    } catch (H5::Exception& e) {
        std::string msg(std::string("an exception occurred while writing model info HDF5 file \n") + e.getCDetailMsg());
        throw StatisticalModelException(msg.c_str());
    }

}

void
BuilderInfo::Load(const H5::CommonFG& modelBuilderGroup) {

    using namespace H5;


    m_modelBuilderName = HDF5Utils::readString(modelBuilderGroup, "./builderName");
    m_buildtime = HDF5Utils::readString(modelBuilderGroup, "./buildTime");

    Group dataInfoGroup = modelBuilderGroup.openGroup("./dataInfo");
    FillKeyValueListFromInfoGroup(dataInfoGroup, m_dataInfo);
    dataInfoGroup.close();

    Group parameterGroup = modelBuilderGroup.openGroup("./parameters");
    FillKeyValueListFromInfoGroup(parameterGroup, m_parameterInfo);
    parameterGroup.close();



}

const BuilderInfo::DataInfoList &BuilderInfo::GetDataInfo() const {
    return m_dataInfo;
}

const BuilderInfo::ParameterInfoList &BuilderInfo::GetParameterInfo() const {
    return m_parameterInfo;
}

inline
void
BuilderInfo::FillKeyValueListFromInfoGroup(const H5::CommonFG& group, KeyValueList& keyValueList) {
    keyValueList.clear();
    unsigned numEntries = group.getNumObjs();
    for (unsigned i = 0; i < numEntries; i++) {
        H5std_string key = group.getObjnameByIdx(i);
        std::string value = HDF5Utils::readString(group, key.c_str());
        keyValueList.push_back(std::make_pair(key, value));
    }
}


} // end namespace
