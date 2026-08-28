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

#ifndef __itkStandardMeshRepresenter_hxx
#define __itkStandardMeshRepresenter_hxx

#include "itkStandardMeshRepresenter.h"

#include <iostream>

#include <itkIdentityTransform.h>
#include <itkMeshFileReader.h>
#include <itkMeshFileWriter.h>
#include <itkPoint.h>
#include <itkTransformMeshFilter.h>
#include <itkVector.h>

#include <boost/filesystem.hpp>

#include "HDF5Utils.h"
#include "StatismoUtils.h"

namespace itk {

template <class TPixel, unsigned MeshDimension>
StandardMeshRepresenter<TPixel, MeshDimension>::StandardMeshRepresenter()
    : m_reference(DatasetType::New()) {
}
template <class TPixel, unsigned MeshDimension>
StandardMeshRepresenter<TPixel, MeshDimension>::~StandardMeshRepresenter() {
}

template <class TPixel, unsigned MeshDimension>
StandardMeshRepresenter<TPixel, MeshDimension>*
StandardMeshRepresenter<TPixel, MeshDimension>::Clone() const {

    StandardMeshRepresenter* clone = new StandardMeshRepresenter();
    clone->Register();

    typename MeshType::Pointer clonedReference = this->CloneDataset(m_reference);
    clone->SetReference(clonedReference);
    return clone;
}


template <class TPixel, unsigned MeshDimension>
void
StandardMeshRepresenter<TPixel, MeshDimension>::Load(const H5::Group& fg) {

    std::string repName = statismo::HDF5Utils::readStringAttribute(fg, "name");
    if (repName == "vtkPolyDataRepresenter" || repName == "itkMeshRepresenter") {
        this->SetReference(LoadRefLegacy(fg));
    } else {
        this->SetReference(LoadRef(fg));
    }
}

template <class TPixel, unsigned MeshDimension>
typename StandardMeshRepresenter<TPixel, MeshDimension>::MeshType::Pointer
StandardMeshRepresenter<TPixel, MeshDimension>::LoadRef(const H5::Group& fg) const {

    statismo::MatrixType vertexMat;
    statismo::HDF5Utils::readMatrix(fg, "./points", vertexMat);

    typedef typename statismo::GenericEigenType<unsigned int>::MatrixType UIntMatrixType;
    UIntMatrixType cellsMat;

    unsigned nVertices = vertexMat.cols();
    unsigned nCells = cellsMat.cols();
    unsigned cellDim = cellsMat.rows();


    typename MeshType::Pointer  mesh = MeshType::New();

    // add points
    for (unsigned i = 0; i < nVertices; i++) {
        typename MeshType::PointType p;
        for(unsigned int j = 0; j < MeshDimension; j++) {
            p[j] = vertexMat(j, i);
        }
        mesh->SetPoint(i, p);
    }

    // add cells
    if(statismo::HDF5Utils::existsObjectWithName(fg, "cells")) {
        statismo::HDF5Utils::readMatrixOfType<unsigned int>(fg, "./cells", cellsMat);

        typedef typename MeshType::CellType::CellAutoPointer         CellAutoPointer;
        typedef itk::LineCell< typename MeshType::CellType >         LineType;
        typedef itk::TriangleCell < typename MeshType::CellType >    TriangleCellType;

        CellAutoPointer cell;

        for (unsigned i = 0; i < nCells; i++) {
            if (cellDim == 2) {
                cell.TakeOwnership(  new LineType  );
            } else if (cellDim == 3) {
                cell.TakeOwnership( new TriangleCellType);
            } else {
                throw statismo::StatisticalModelException("This representer currently supports only line and triangle cells");
            }

            for (unsigned d = 0; d < cellDim; d++) {
                cell->SetPointId(d, cellsMat(d, i));
            }
            mesh->SetCell( i, cell );
        }
    }


    // currently this representer supports only pointdata of type scalar
    if (statismo::HDF5Utils::existsObjectWithName(fg, "pointData")) {
        H5::Group pdGroup = fg.openGroup("./pointData");

        if (statismo::HDF5Utils::existsObjectWithName(pdGroup, "scalars")) {
            H5::DataSet ds = pdGroup.openDataSet("scalars");
            unsigned type = static_cast<unsigned>(statismo::HDF5Utils::readIntAttribute(ds, "datatype"));
            if (type != PixelConversionTrait<TPixel>::GetDataType()) {
                std::cout << "Warning: The datatype specified for the scalars does not match the TPixel template argument used in this representer." << std::endl;
            }
            statismo::MatrixTypeDoublePrecision scalarMatDouble;
            statismo::HDF5Utils::readMatrixOfType<double>(pdGroup, "scalars", scalarMatDouble);
            statismo::MatrixType scalarMat = scalarMatDouble.cast<statismo::ScalarType>();
            assert(static_cast<unsigned>(scalarMatDouble.cols()) == mesh->GetNumberOfPoints());
            typename MeshType::PointDataContainerPointer pd = MeshType::PointDataContainer::New();

            for (unsigned i = 0; i < scalarMatDouble.cols(); i++) {
                TPixel v = PixelConversionTrait<TPixel>::FromVector(scalarMat.col(i));
                pd->InsertElement(i, v);
            }
            mesh->SetPointData(pd);
        }

        pdGroup.close();
    }

    return mesh;
}


template <class TPixel, unsigned MeshDimension>
typename StandardMeshRepresenter<TPixel, MeshDimension>::MeshType::Pointer
StandardMeshRepresenter<TPixel, MeshDimension>::LoadRefLegacy(const H5::Group& fg) const {

    std::string tmpfilename = statismo::Utils::CreateTmpName(".vtk");
    statismo::HDF5Utils::getFileFromHDF5(fg, "./reference", tmpfilename.c_str());


    typename itk::MeshFileReader<MeshType>::Pointer reader = itk::MeshFileReader<MeshType>::New();
    reader->SetFileName(tmpfilename);
    try {
        reader->Update();
    } catch (itk::MeshFileReaderException& e) {
        boost::filesystem::remove(tmpfilename);
        throw statismo::StatisticalModelException((std::string("Could not read file ") + tmpfilename).c_str());
    }

    typename MeshType::Pointer mesh = reader->GetOutput();
    boost::filesystem::remove(tmpfilename);
    return mesh;

}


template <class TPixel, unsigned MeshDimension>
void
StandardMeshRepresenter<TPixel, MeshDimension>::SetReference(DatasetPointerType reference) {
    m_reference = reference;

    // We create a list of poitns for the domain.
    // Furthermore, we cache for all the points of the reference, as these are the most likely ones
    // we have to look up later.
    typename DomainType::DomainPointsListType domainPointList;

    typename PointsContainerType::ConstPointer points = m_reference->GetPoints();
    typename PointsContainerType::ConstIterator pointIterator = points->Begin();
    unsigned id = 0;
    while( pointIterator != points->End() ) {
        domainPointList.push_back(pointIterator.Value());
        m_pointCache.insert(std::pair<PointType, unsigned>(pointIterator.Value(), id));
        ++pointIterator;
        ++id;
    }
    m_domain = DomainType(domainPointList);

}

template <class TPixel, unsigned MeshDimension>
statismo::VectorType
StandardMeshRepresenter<TPixel, MeshDimension>::PointToVector(const PointType& pt) const {
    statismo::VectorType v(PointType::GetPointDimension());
    for (unsigned i = 0; i < PointType::GetPointDimension(); i++) {
        v(i) = pt[i];
    }
    return v;

}

template <class TPixel, unsigned MeshDimension>
statismo::VectorType
StandardMeshRepresenter<TPixel, MeshDimension>::SampleToSampleVector(DatasetConstPointerType mesh) const {
    statismo::VectorType sample(GetNumberOfPoints() * GetDimensions());

    typename PointsContainerType::ConstPointer points = mesh->GetPoints();

    typename PointsContainerType::ConstIterator pointIterator = points->Begin();
    unsigned id = 0;
    while( pointIterator != points->End() ) {
        for (unsigned d = 0; d < GetDimensions(); d++) {
            unsigned idx = this->MapPointIdToInternalIdx(id, d);
            sample[idx] = pointIterator.Value()[d];
        }
        ++pointIterator;
        ++id;
    }
    return sample;
}



template <class TPixel, unsigned MeshDimension>
typename StandardMeshRepresenter<TPixel, MeshDimension>::DatasetPointerType
StandardMeshRepresenter<TPixel, MeshDimension>::SampleVectorToSample(const statismo::VectorType& sample) const {
    typename MeshType::Pointer mesh = this->CloneDataset(m_reference);
    typename PointsContainerType::Pointer points = mesh->GetPoints();
    typename PointsContainerType::Iterator pointsIterator = points->Begin();

    unsigned ptId = 0;
    while( pointsIterator != points->End() ) {
        ValueType v;
        for (unsigned d = 0; d < GetDimensions(); d++) {
            unsigned idx = this->MapPointIdToInternalIdx(ptId, d);
            v[d] = sample[idx];
        }
        mesh->SetPoint(ptId, v);

        ++ptId;
        ++pointsIterator;
    }
    return mesh;
}

template <class TPixel, unsigned MeshDimension>
typename StandardMeshRepresenter<TPixel, MeshDimension>::ValueType
StandardMeshRepresenter<TPixel, MeshDimension>::PointSampleFromSample(DatasetConstPointerType sample, unsigned ptid) const {
    if (ptid >= sample->GetNumberOfPoints()) {
        throw statismo::StatisticalModelException("invalid ptid provided to PointSampleFromSample");
    }

    return sample->GetPoint(ptid);
}


template <class TPixel, unsigned MeshDimension>
typename StandardMeshRepresenter<TPixel, MeshDimension>::ValueType
StandardMeshRepresenter<TPixel, MeshDimension>::PointSampleVectorToPointSample(const statismo::VectorType& pointSample) const {
    ValueType value;
    for (unsigned d = 0; d < GetDimensions(); d++) {
        value[d] = pointSample[d];
    }
    return value;
}
template <class TPixel, unsigned MeshDimension>
statismo::VectorType
StandardMeshRepresenter<TPixel, MeshDimension>::PointSampleToPointSampleVector(const ValueType& v) const {
    statismo::VectorType vec(GetDimensions());
    for (unsigned d = 0; d < GetDimensions(); d++) {
        vec[d] = v[d];
    }
    return vec;
}


template <class TPixel, unsigned MeshDimension>
void
StandardMeshRepresenter<TPixel, MeshDimension>::Save(const H5::Group& fg) const {
    using namespace H5;

    statismo::MatrixType vertexMat = statismo::MatrixType::Zero(3, m_reference->GetNumberOfPoints());

    for (unsigned i = 0; i < m_reference->GetNumberOfPoints(); i++) {
        typename MeshType::PointType pt = m_reference->GetPoint(i);
        for (unsigned d = 0; d < 3; d++) {
            vertexMat(d, i) = pt[d];
        }
    }
    statismo::HDF5Utils::writeMatrix(fg, "./points", vertexMat);

    H5::Group pdGroup = fg.createGroup("pointData");

    typename MeshType::PointDataContainerConstPointer pd = m_reference->GetPointData();
    if (pd.IsNotNull() && pd->Size() == m_reference->GetNumberOfPoints()) {
        unsigned numComponents = PixelConversionTrait<TPixel>::ToVector(pd->GetElement(0)).rows();

        statismo::MatrixType scalarsMat = statismo::MatrixType::Zero(numComponents, m_reference->GetNumberOfPoints());
        for (unsigned i = 0; i < m_reference->GetNumberOfPoints(); i++) {
            scalarsMat.col(i) = PixelConversionTrait<TPixel>::ToVector(pd->GetElement(i));
        }
        statismo::MatrixTypeDoublePrecision scalarsMatDouble = scalarsMat.cast<double>();
        H5::DataSet ds = statismo::HDF5Utils::writeMatrixOfType<double>(pdGroup, "scalars", scalarsMatDouble);
        statismo::HDF5Utils::writeIntAttribute(ds, "datatype", PixelConversionTrait<TPixel>::GetDataType());
    }

    if(this->m_reference->GetNumberOfCells()) {
        // check the dimensionality of a face (i.e. the number of points it has). We assume that
        // all the cells are the same.
        unsigned numPointsPerCell = 0;
        if (m_reference->GetNumberOfCells() > 0) {
            typename MeshType::CellAutoPointer cellPtr;
            m_reference->GetCell(0, cellPtr);
            numPointsPerCell = cellPtr->GetNumberOfPoints();
        }

        typedef typename statismo::GenericEigenType<unsigned int>::MatrixType UIntMatrixType;
        UIntMatrixType facesMat = UIntMatrixType::Zero(numPointsPerCell, m_reference->GetNumberOfCells());


        for (unsigned i = 0; i < m_reference->GetNumberOfCells(); i++) {
            typename MeshType::CellAutoPointer cellPtr;
            m_reference->GetCell(i, cellPtr);
            assert(numPointsPerCell == cellPtr->GetNumberOfPoints());
            for (unsigned d = 0; d < numPointsPerCell; d++) {
                facesMat(d, i) = cellPtr->GetPointIds()[d];
            }
        }

        statismo::HDF5Utils::writeMatrixOfType<unsigned int>(fg, "./cells", facesMat);
    }
}


template <class TPixel, unsigned MeshDimension>
unsigned
StandardMeshRepresenter<TPixel, MeshDimension>::GetNumberOfPoints() const {
    return this->m_reference->GetNumberOfPoints();
}


template <class TPixel, unsigned MeshDimension>
unsigned
StandardMeshRepresenter<TPixel, MeshDimension>::GetPointIdForPoint(const PointType& pt) const {
    int ptId = -1;

    // check whether the point is cached, otherwise look for it
    typename PointCacheType::const_iterator got = m_pointCache.find (pt);
    if (got == m_pointCache.end()) {
        ptId = FindClosestPoint(m_reference, pt);
        m_pointCache.insert(std::pair<PointType, unsigned>(pt, ptId));
    } else {
        ptId = got->second;
    }
    assert(ptId != -1);
    return static_cast<unsigned>(ptId);
}



template <class TPixel, unsigned MeshDimension>
typename  StandardMeshRepresenter<TPixel, MeshDimension>::DatasetPointerType
StandardMeshRepresenter<TPixel, MeshDimension>::CloneDataset(DatasetConstPointerType mesh) const {

    // cloning is cumbersome - therefore we let itk do the job for, and use perform a
    // Mesh transform using the identity transform. This should result in a perfect clone.

    typedef itk::IdentityTransform<TPixel, MeshDimension> IdentityTransformType;
    typedef itk::TransformMeshFilter<MeshType, MeshType, IdentityTransformType> TransformMeshFilterType;

    typename TransformMeshFilterType::Pointer tf = TransformMeshFilterType::New();
    tf->SetInput(mesh);
    typename IdentityTransformType::Pointer idTrans = IdentityTransformType::New();
    tf->SetTransform(idTrans);
    tf->Update();

    typename MeshType::Pointer clone = tf->GetOutput();
    clone->DisconnectPipeline();
    return clone;
}

template <class TPixel, unsigned MeshDimension>
unsigned
StandardMeshRepresenter<TPixel, MeshDimension>::FindClosestPoint(const MeshType* mesh, const PointType pt) const {
    throw statismo::StatisticalModelException("Not implemented. Currently only points of the reference can be used.");
}

} // namespace itk

#endif
