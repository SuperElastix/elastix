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
 * PROFITS; OR BUSINESS addINTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <iostream>

#include "CommonTypes.h"
#include "HDF5Utils.h"
#include "StatismoUtils.h"

/**
 * This class provides generic tests for representer. The tests need to hold for all representers.
 */
template <typename Representer>
class GenericRepresenterTest {


    // we define typedefs for all required typenames, to force a compilation error if one of them
    // is not defined
    typedef typename Representer::DatasetConstPointerType DatasetConstPointerType;
    typedef typename Representer::DatasetPointerType DatasetPointerType;
    typedef typename Representer::PointType PointType;
    typedef typename Representer::ValueType ValueType;

    typedef typename Representer::DomainType DomainType;

  public:

    /// Create new test with the given representer.
    /// Tests are performed using the given testDataset and the pointValuePair.
    /// It is assumed that the PointValuePair is taken from the testDataset (otherwise some tests will fail).
    GenericRepresenterTest(const Representer* representer, DatasetConstPointerType testDataset, std::pair<PointType, ValueType> pointValuePair)
        : m_representer(representer),
          m_testDataset(testDataset),
          m_testPoint(pointValuePair.first),
          m_testValue(pointValuePair.second) {
    }


    bool testSamplePointEvaluation() const {
        DatasetConstPointerType sample = m_testDataset;
        unsigned id = m_representer->GetPointIdForPoint(m_testPoint);
        ValueType val = m_representer->PointSampleFromSample(sample, id);
        statismo::VectorType valVec = m_representer->PointSampleToPointSampleVector(val);

        // the obtained value should correspond to the value that is obtained by obtaining the sample vector, and evaluating it at the given position
        statismo::VectorType sampleVector = m_representer->SampleToSampleVector(sample);
        for (unsigned i = 0; i < m_representer->GetDimensions(); ++i) {
            unsigned idx = m_representer->MapPointIdToInternalIdx(id, i);
            if (sampleVector(i) != valVec(i)) {
                return false;
            }
        }
        return true;
    }

    bool testDomainValid() const {
        std::cout << "testDomainValid" << std::endl;

        const DomainType domain = m_representer->GetDomain();
        typename DomainType::DomainPointsListType domPoints = domain.GetDomainPoints();

        if (domPoints.size() == 0) {
            std::cout << "representer defined empty domain" << std::endl;
            return false;
        }
        if (domPoints.size() != domain.GetNumberOfPoints()) {
            std::cout << "domPoints.size() != domain.GetNumberOfPoints() (" << domPoints.size() << " != " << domain.GetNumberOfPoints() << std::endl;
            return false;
        }
        // if we convert a dataset to a samplevector, the resulting vector needs to have
        // as many entries as there are points * dimensions
        DatasetConstPointerType sample = m_testDataset;
        statismo::VectorType sampleVector = m_representer->SampleToSampleVector(sample);
        if (sampleVector.rows() != m_representer->GetDimensions() * domain.GetNumberOfPoints()) {
            std::cout << "the dimension of the sampleVector does not agree with the number of points in the domain (#points * dimensionality)" << std::endl;
            return false;
        }


        unsigned ptNo = 0;
        for (typename DomainType::DomainPointsListType::const_iterator it = domPoints.begin();
                it != domPoints.end();
                ++it) {
            // since this can take long, we only do it for every 10th point
            if (ptNo % 10 != 0)
                break;

            if (m_representer->GetPointIdForPoint(*it) >= domain.GetNumberOfPoints()) {
                std::cout << "a point in the domain did not evaluate to a valid point it" << std::endl;
                return false;
            }
            ptNo++;
        }


        return true;

    }



    /// test whether converting a sample to a vector and back to a sample yields the original sample
    bool testSampleToVectorAndBack() const {
        std::cout << "testSampleToVectorToSample" << std::endl;

        statismo::VectorType sampleVec = getSampleVectorFromTestDataset();

        DatasetConstPointerType reconstructedSample = m_representer->SampleVectorToSample(sampleVec);

        // as we don't know anything about how to compare samples, we compare their vectorial representation
        statismo::VectorType reconstructedSampleAsVec = m_representer->SampleToSampleVector(reconstructedSample);
        bool isOkay = assertSampleVectorsEqual(sampleVec, reconstructedSampleAsVec);
        if (isOkay == false) {
            std::cout << "Error: the sample has changed by converting between the representations " << std::endl;
        }
        return true;
    }

    /// test if the pointSamples have the correct dimensionality
    bool testPointSampleDimension() const {
        std::cout << "testPointSampleDimension" << std::endl;

        statismo::VectorType valVec = m_representer->PointSampleToPointSampleVector(m_testValue);

        if (valVec.rows() != m_representer->GetDimensions()) {
            std::cout << "Error: The dimensionality of the pointSampleVector is not the same as the Dimensionality of the representer" << std::endl;
            return false;
        }
        return true;
    }


    /// tests if the conversion from a pointSample and the pointSampleVector, and back to a pointSample
    /// yields the original sample.
    bool testPointSampleToPointSampleVectorAndBack() const {
        std::cout << "testPointSampleToPointSampleVectorAndBack" << std::endl;

        statismo::VectorType valVec = m_representer->PointSampleToPointSampleVector(m_testValue);
        ValueType recVal = m_representer->PointSampleVectorToPointSample(valVec);

        // we compare the vectors and not the points, as we don't know how to compare poitns.
        statismo::VectorType recValVec = m_representer->PointSampleToPointSampleVector(recVal);
        bool ok = assertSampleVectorsEqual(valVec, recValVec);
        if (!ok) {
            std::cout << "Error: the point sample has changed by converting between the representations" << std::endl;
        }
        return ok;
    }

    /// test if the testSample contains the same entries in the vector as those obtained by taking the
    /// pointSample at the corresponding position.
    bool testSampleVectorHasCorrectValueAtPoint() const {
        std::cout << "testSampleVectorHasCorrectValueAtPoint" << std::endl;

        unsigned ptId = m_representer->GetPointIdForPoint(m_testPoint);
        if (ptId < 0 || ptId >= m_representer->GetNumberOfPoints()) {
            std::cout << "Error: invalid point id for test point " << ptId << std::endl;
            return false;
        }

        // the value of the point in the sample vector needs to correspond the the value that was provided
        statismo::VectorType sampleVec = getSampleVectorFromTestDataset();
        statismo::VectorType pointSampleVec = m_representer->PointSampleToPointSampleVector(m_testValue);

        for (unsigned d = 0; d < m_representer->GetDimensions(); ++d) {
            unsigned idx = m_representer->MapPointIdToInternalIdx(ptId, d);
            if (sampleVec[idx] != pointSampleVec[d]) {
                std::cout << "Error: the sample vector does not contain the correct value of the pointSample " << std::endl;
                return false;
            }
        }
        return true;

    }


    /// test whether the representer is correctly restored
    bool testSaveLoad() const {
        std::cout << "testSaveLoad" << std::endl;

        using namespace H5;

        std::string filename = statismo::Utils::CreateTmpName(".rep");
        H5File file;
        try {
            file = H5File( filename, H5F_ACC_TRUNC );
        } catch (Exception& e) {
            std::string msg(std::string("Error: Could not open HDF5 file for writing \n") + e.getCDetailMsg());
            std::cout << msg << std::endl;
            return false;
        }
        H5::Group representerGroup = file.createGroup("/representer");

        m_representer->Save(representerGroup);

        // We add the required attributes, which are usually written by the StatisticalModel class.
        // This is needed, as some representers check on these values.

        statismo::HDF5Utils::writeStringAttribute(representerGroup, "name", m_representer->GetName());
        std::string dataTypeStr = Representer::TypeToString(m_representer->GetType());
        statismo::HDF5Utils::writeStringAttribute(representerGroup, "datasetType", dataTypeStr);

        file.close();
        try {
            file = H5File(filename.c_str(), H5F_ACC_RDONLY);
        } catch (Exception& e) {
            std::string msg(std::string("Error: could not open HDF5 file \n") + e.getCDetailMsg());
            std::cout << msg << std::endl;
            return false;
        }

        representerGroup.close();
        representerGroup = file.openGroup("/representer");

        Representer* newRep = Representer::Create();
        newRep->Load(representerGroup);

        bool isOkay = assertRepresenterEqual(newRep, m_representer);
        newRep->Delete();

        return isOkay;

    }

    /// test whether cloning a representer results in a representer with the same behaviour
    bool testClone() const {
        std::cout << "testClone" << std::endl;
        bool isOkay = true;

        Representer* rep = m_representer->Clone();
        if (assertRepresenterEqual(rep, m_representer) == false) {
            std::cout << "Error: the clone of the representer is not the same as the representer " << std::endl;
            isOkay = false;
        }
        rep->Delete();
        return isOkay;
    }

    /// test if the sample vector dimensions are correct
    bool testSampleVectorDimensions() const {
        std::cout << "testSampleVectorDimensions()" << std::endl;
        statismo::VectorType testSampleVec = getSampleVectorFromTestDataset();

        bool isOk =  m_representer->GetDimensions() * m_representer->GetNumberOfPoints() == testSampleVec.rows();
        if (!isOk) {
            std::cout << "Error: Dimensionality of the sample vector does not agree with the representer parameters "
                      << "dimension and numberOfPoints" << std::endl;
            std::cout << testSampleVec.rows() << " != " << m_representer->GetDimensions() << " * " << m_representer->GetNumberOfPoints() << std::endl;
        }
        return isOk;
    }

    /// test whether the name is defined
    bool testGetName() const {
        std::cout << "testGetName" << std::endl;

        if (m_representer->GetName() == "") {
            std::cout << "Error: representer name has to be non empty" << std::endl;
            return false;
        }
        return true;
    }

    /// test if the dimensionality is nonnegative
    bool testDimensions() const {
        std::cout << "testDimensions " << std::endl;

        if (m_representer->GetDimensions() <= 0)  {
            std::cout << "Error: Dimensionality of representer has to be > 0" << std::endl;
            return false;
        }
        return true;
    }

    /// run all the tests
    bool runAllTests() {
        bool ok = true;
        ok = testPointSampleDimension() && ok;
        ok = testSamplePointEvaluation() && ok;
        ok = testDomainValid() && ok;
        ok = testPointSampleToPointSampleVectorAndBack() && ok;
        ok = testSampleVectorHasCorrectValueAtPoint() && ok;
        ok = testSampleToVectorAndBack() && ok;
        ok = testSaveLoad() && ok;
        ok = testClone() && ok;
        ok = testSampleVectorDimensions() && ok;
        ok = testGetName() && ok;
        ok = testDimensions() && ok;
        return ok;
    }


  private:


    bool assertRepresenterEqual(const Representer* representer1, const Representer* representer2) const {
        if (representer1->GetNumberOfPoints() != representer2->GetNumberOfPoints()) {
            std::cout << "the representers do not have the same nubmer of points " <<std::endl;
            return false;
        }
        statismo::VectorType sampleRep1 = getSampleVectorFromTestDataset(representer1);
        statismo::VectorType sampleRep2 = getSampleVectorFromTestDataset(representer2);
        if (assertSampleVectorsEqual(sampleRep1, sampleRep2) == false) {
            std::cout << "the representers produce different sample vectors for the same sample" << std::endl;
            return false;
        }

        return true;
    }


    bool assertSampleVectorsEqual(const statismo::VectorType& v1, const statismo::VectorType& v2) const {
        if (v1.rows() != v2.rows()) {
            std::cout << "dimensionality of SampleVectors do not agree" << std::endl;
            return false;
        }

        for (unsigned i = 0; i < v1.rows(); ++i) {
            if (v1[i] != v2[i]) {
                std::cout << "the sample vectors are not the same" << std::endl;
                return false;
            }
        }

        return true;
    }


    statismo::VectorType getSampleVectorFromTestDataset() const {
        return getSampleVectorFromTestDataset(m_representer);
    }

    statismo::VectorType getSampleVectorFromTestDataset(const Representer* representer) const {
        DatasetConstPointerType sample = m_testDataset;
        statismo::VectorType sampleVec = representer->SampleToSampleVector(sample);
        return sampleVec;
    }


    const Representer* m_representer;
    DatasetConstPointerType m_testDataset;
    PointType m_testPoint;
    ValueType m_testValue;

};


