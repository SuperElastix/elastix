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

#ifndef __itkStandardImageRepresenter_hxx
#define __itkStandardImageRepresenter_hxx


#include <iostream>

#include <itkImageDuplicator.h>
#include <itkImageIterator.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkIndex.h>
#include <itkPoint.h>
#include <itkVector.h>

#include <boost/filesystem.hpp>

#include "HDF5Utils.h"
#include "StatismoUtils.h"

#include "itkStandardImageRepresenter.h"


namespace itk {

template <class TPixel, unsigned ImageDimension>
StandardImageRepresenter<TPixel, ImageDimension>::StandardImageRepresenter()
    : m_reference(0) {
}
template <class TPixel, unsigned ImageDimension>
StandardImageRepresenter<TPixel, ImageDimension>::~StandardImageRepresenter() {
}

template <class TPixel, unsigned ImageDimension>
StandardImageRepresenter<TPixel, ImageDimension>*
StandardImageRepresenter<TPixel, ImageDimension>::Clone() const {

    StandardImageRepresenter* clone = new StandardImageRepresenter();
    clone->Register();

    DatasetPointerType clonedReference = this->CloneDataset(m_reference);
    clone->SetReference(clonedReference);
    return clone;
}



template <class TPixel, unsigned ImageDimension>
void
StandardImageRepresenter<TPixel, ImageDimension>::Load(const H5::Group& fg) {

    std::string repName = statismo::HDF5Utils::readStringAttribute(fg, "name");
    if (repName == "vtkStructuredPointsRepresenter" || repName == "itkImageRepresenter" || repName == "itkVectorImageRepresenter") {
        this->SetReference(LoadRefLegacy(fg));
    } else {
        this->SetReference(LoadRef(fg));
    }

}


template <class TPixel, unsigned ImageDimension>
typename StandardImageRepresenter<TPixel, ImageDimension>::ImageType::Pointer
StandardImageRepresenter<TPixel, ImageDimension>::LoadRef(const H5::Group& fg) const {


    int readImageDimension = statismo::HDF5Utils::readInt(fg, "imageDimension");
    if (readImageDimension != ImageDimension)  {
        throw statismo::StatisticalModelException("the image dimension specified in the statismo file does not match the one specified as template parameter");
    }


    statismo::VectorType originVec;
    statismo::HDF5Utils::readVector(fg, "origin", originVec);
    typename ImageType::PointType origin;
    for (unsigned i = 0; i < ImageDimension; i++) {
        origin[i] = originVec[i];
    }

    statismo::VectorType spacingVec;
    statismo::HDF5Utils::readVector(fg, "spacing", spacingVec);
    typename ImageType::SpacingType spacing;
    for (unsigned i = 0; i < ImageDimension; i++) {
        spacing[i] = spacingVec[i];
    }

    typename statismo::GenericEigenType<int>::VectorType sizeVec;
    statismo::HDF5Utils::readVectorOfType<int>(fg, "size", sizeVec);
    typename ImageType::SizeType size;
    for (unsigned i = 0; i < ImageDimension; i++) {
        size[i] = sizeVec[i];
    }

    statismo::MatrixType directionMat;
    statismo::HDF5Utils::readMatrix(fg, "direction", directionMat);
    typename ImageType::DirectionType direction;
    for (unsigned i = 0; i < directionMat.rows(); i++) {
        for (unsigned j = 0; j < directionMat.rows(); j++) {
            direction[i][j] = directionMat(i,j);
        }
    }

    H5::Group pdGroup = fg.openGroup("./pointData");
    unsigned readPixelDimension = static_cast<unsigned>(statismo::HDF5Utils::readInt(pdGroup, "pixelDimension"));
    if (readPixelDimension != GetDimensions())  {
        throw statismo::StatisticalModelException("the pixel dimension specified in the statismo file does not match the one specified as template parameter");
    }

    typename statismo::GenericEigenType<double>::MatrixType pixelMatDouble;
    statismo::HDF5Utils::readMatrixOfType<double>(pdGroup, "pixelValues", pixelMatDouble);
    statismo::MatrixType pixelMat = pixelMatDouble.cast<statismo::ScalarType>();
    typename ImageType::Pointer newImage = ImageType::New();
    typename ImageType::IndexType start;
    start.Fill(0);


    H5::DataSet ds = pdGroup.openDataSet("pixelValues");
    unsigned int type = static_cast<unsigned>(statismo::HDF5Utils::readIntAttribute(ds, "datatype"));
    if (type != PixelConversionTrait<TPixel>::GetDataType()) {
        std::cout << "Warning: The datatype specified for the scalars does not match the TPixel template argument used in this representer." << std::endl;
    }
    pdGroup.close();
    typename ImageType::RegionType region(start, size);
    newImage->SetRegions(region);
    newImage->Allocate();
    newImage->SetOrigin(origin);
    newImage->SetSpacing(spacing);
    newImage->SetDirection(direction);


    itk::ImageRegionIterator<DatasetType> it(newImage, newImage->GetLargestPossibleRegion());
    it.GoToBegin();
    for (unsigned i  = 0;  !it.IsAtEnd(); ++it, i++) {
        TPixel v = PixelConversionTrait<TPixel>::FromVector(pixelMat.col(i));
        it.Set(v);
    }

    return newImage;
}

template <class TPixel, unsigned ImageDimension>
typename StandardImageRepresenter<TPixel, ImageDimension>::ImageType::Pointer
StandardImageRepresenter<TPixel, ImageDimension>::LoadRefLegacy(const H5::Group& fg) const {

    std::string tmpfilename;
    tmpfilename = statismo::Utils::CreateTmpName(".vtk");
    statismo::HDF5Utils::getFileFromHDF5(fg, "./reference", tmpfilename.c_str());

    typename itk::ImageFileReader<ImageType>::Pointer reader = itk::ImageFileReader<ImageType>::New();
    reader->SetFileName(tmpfilename);
    try {
        reader->Update();
    } catch (itk::ImageFileReaderException& e) {
        boost::filesystem::remove(tmpfilename);
        throw statismo::StatisticalModelException((std::string("Could not read file ") + tmpfilename).c_str());
    }
    typename DatasetType::Pointer img = reader->GetOutput();
    img->Register();
    boost::filesystem::remove(tmpfilename);
    return img;

}


template <class TPixel, unsigned ImageDimension>
void
StandardImageRepresenter<TPixel, ImageDimension>::SetReference(ImageType* reference) {
    m_reference = reference;

    typename DomainType::DomainPointsListType domainPoints;
    itk::ImageRegionConstIterator<DatasetType> it(reference, reference->GetLargestPossibleRegion());
    it.GoToBegin();
    for (;
            it.IsAtEnd() == false
            ;) {
        PointType pt;
        reference->TransformIndexToPhysicalPoint(it.GetIndex(), pt);
        domainPoints.push_back(pt);
        ++it;
    }
    m_domain = DomainType(domainPoints);
}

template <class TPixel, unsigned ImageDimension>
statismo::VectorType
StandardImageRepresenter<TPixel, ImageDimension>::PointToVector(const PointType& pt) const {
    statismo::VectorType v(PointType::GetPointDimension());
    for (unsigned i = 0; i < PointType::GetPointDimension(); i++) {
        v(i) = pt[i];
    }
    return v;

}




template <class TPixel, unsigned ImageDimension>
statismo::VectorType
StandardImageRepresenter<TPixel, ImageDimension>::SampleToSampleVector(DatasetConstPointerType image) const {
    statismo::VectorType sample(this->GetNumberOfPoints() * GetDimensions());
    itk::ImageRegionConstIterator<DatasetType> it(image, image->GetLargestPossibleRegion());

    it.GoToBegin();
    for (unsigned i = 0;
            it.IsAtEnd() == false;
            ++i) {

        statismo::VectorType sampleAtPt =  PixelConversionTrait<TPixel>::ToVector(it.Value());
        for (unsigned j = 0; j < GetDimensions(); j++) {
            unsigned idx = this->MapPointIdToInternalIdx(i, j);
            sample[idx] = sampleAtPt[j];
        }
        ++it;
    }
    return sample;
}


template <class TPixel, unsigned ImageDimension>
typename StandardImageRepresenter<TPixel, ImageDimension>::DatasetPointerType
StandardImageRepresenter<TPixel, ImageDimension>::SampleVectorToSample(const statismo::VectorType& sample) const {

    typedef itk::ImageDuplicator< DatasetType > DuplicatorType;
    typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
    duplicator->SetInputImage(this->m_reference);
    duplicator->Update();
    DatasetPointerType clonedImage = duplicator->GetOutput();

    itk::ImageRegionIterator<DatasetType> it(clonedImage, clonedImage->GetLargestPossibleRegion());
    it.GoToBegin();
    for (unsigned i  = 0;  !it.IsAtEnd(); ++it, i++) {

        statismo::VectorType valAtPoint(GetDimensions());
        for (unsigned d = 0; d < GetDimensions(); d++) {
            unsigned idx = this->MapPointIdToInternalIdx(i, d);
            valAtPoint[d] = sample[idx];
        }
        ValueType v = PixelConversionTrait<TPixel>::FromVector(valAtPoint);
        it.Set(v);
    }
    return clonedImage;

}

template <class TPixel, unsigned ImageDimension>
typename StandardImageRepresenter<TPixel, ImageDimension>::ValueType
StandardImageRepresenter<TPixel, ImageDimension>::PointSampleFromSample(DatasetConstPointerType sample, unsigned ptid) const {
    if (ptid >= GetDomain().GetNumberOfPoints()) {
        throw statismo::StatisticalModelException("invalid ptid provided to PointSampleFromSample");
    }

    // we get the point with the id from the domain, as itk does not allow us get a point via its index.
    PointType pt = GetDomain().GetDomainPoints()[ptid];
    typename ImageType::IndexType idx;
    sample->TransformPhysicalPointToIndex(pt, idx);
    ValueType value = sample->GetPixel(idx);
    return value;

}

template <class TPixel, unsigned ImageDimension>
typename StandardImageRepresenter<TPixel, ImageDimension>::ValueType
StandardImageRepresenter<TPixel, ImageDimension>::PointSampleVectorToPointSample(const statismo::VectorType& pointSample) const {
    return PixelConversionTrait<TPixel>::FromVector(pointSample);
}

template <class TPixel, unsigned ImageDimension>
statismo::VectorType
StandardImageRepresenter<TPixel, ImageDimension>::PointSampleToPointSampleVector(const ValueType& v) const {
    return PixelConversionTrait<TPixel>::ToVector(v);
}


template <class TPixel, unsigned ImageDimension>
void
StandardImageRepresenter<TPixel, ImageDimension>::Save(const H5::Group& fg) const {

    typename ImageType::PointType origin = m_reference->GetOrigin();
    statismo::VectorType originVec(ImageDimension);
    for (unsigned i = 0; i < ImageDimension; i++) {
        originVec(i) = origin[i];
    }
    statismo::HDF5Utils::writeVector(fg, "origin", originVec);

    typename ImageType::SpacingType spacing = m_reference->GetSpacing();
    statismo::VectorType spacingVec(ImageDimension);
    for (unsigned i = 0; i < ImageDimension; i++) {
        spacingVec(i) = spacing[i];
    }
    statismo::HDF5Utils::writeVector(fg, "spacing", spacingVec);


    statismo::GenericEigenType<int>::VectorType sizeVec(ImageDimension);
    for (unsigned i = 0; i < ImageDimension; i++) {
        sizeVec(i) = m_reference->GetLargestPossibleRegion().GetSize()[i];
    }
    statismo::HDF5Utils::writeVectorOfType<int>(fg, "size", sizeVec);

    typename ImageType::DirectionType direction = m_reference->GetDirection();
    statismo::MatrixType directionMat(ImageDimension, ImageDimension);
    for (unsigned i = 0; i < ImageDimension; i++) {
        for (unsigned j = 0; j < ImageDimension; j++) {
            directionMat(i,j) = direction[i][j];
        }
    }
    statismo::HDF5Utils::writeMatrix(fg, "direction", directionMat);

    statismo::HDF5Utils::writeInt(fg, "imageDimension", ImageDimension);

    H5::Group pdGroup = fg.createGroup("pointData");
    statismo::HDF5Utils::writeInt(pdGroup, "pixelDimension", GetDimensions());


    typedef statismo::GenericEigenType<double>::MatrixType DoubleMatrixType;
    statismo::MatrixType pixelMat(GetDimensions(), GetNumberOfPoints());

    itk::ImageRegionIterator<DatasetType> it(m_reference, m_reference->GetLargestPossibleRegion());
    it.GoToBegin();
    for (unsigned i = 0;
            it.IsAtEnd() == false;
            ++i) {
        pixelMat.col(i) = PixelConversionTrait<TPixel>::ToVector(it.Get());
        ++it;
    }
    DoubleMatrixType pixelMatDouble = pixelMat.cast<double>();
    H5::DataSet ds = statismo::HDF5Utils::writeMatrixOfType<double>(pdGroup, "pixelValues", pixelMatDouble);
    statismo::HDF5Utils::writeIntAttribute(ds, "datatype", PixelConversionTrait<TPixel>::GetDataType());
    pdGroup.close();
}


template <class TPixel, unsigned ImageDimension>
unsigned
StandardImageRepresenter<TPixel, ImageDimension>::GetNumberOfPoints() const {
    return m_reference->GetLargestPossibleRegion().GetNumberOfPixels();
}


template <class TPixel, unsigned ImageDimension>
unsigned
StandardImageRepresenter<TPixel, ImageDimension>::GetPointIdForPoint(const PointType& pt) const {
    // itks organization is slice row col
    typename DatasetType::IndexType idx;
    bool ptInImage = this->m_reference->TransformPhysicalPointToIndex(pt, idx);

    typename DatasetType::SizeType size = this->m_reference->GetLargestPossibleRegion().GetSize();

    // It does not make sense to allow points outside the image, because only the inside is modeled.
    // However, some discretization artifacts of image and surface operations may produce points that
    // are just on the boundary of the image, but mathematically outside. We accept these points and
    // return the iD of the closest image point.
    // Any points further out will trigger an exception.
    if(!ptInImage) {
        for (unsigned int i=0; i<ImageType::ImageDimension; ++i) {
            // As soon as one coordinate is further away than one pixel, we throw an exception.
            if(idx[i] < -1 || idx[i] > size[i]) {
                throw statismo::StatisticalModelException("GetPointIdForPoint computed invalid ptId. Make sure that the point is within the reference you chose ");
            }
            // If it is on the boundary, we set it to the nearest boundary coordinate.
            if(idx[i] == -1) idx[i] = 0;
            if(idx[i] == size[i]) idx[i] = size[i] - 1;
        }
    }


    // in itk, idx 0 is by convention the fastest moving index
    unsigned int index=0;
    for (unsigned int i=0; i<ImageType::ImageDimension; ++i) {
        unsigned int multiplier=1;
        for (int d=i-1; d>=0; --d) {
            multiplier*=size[d];
        }
        index+=multiplier*idx[i];
    }

    return index;
}

template <class TPixel, unsigned ImageDimension>
typename StandardImageRepresenter<TPixel, ImageDimension>::DatasetPointerType
StandardImageRepresenter<TPixel, ImageDimension>::CloneDataset(DatasetConstPointerType d) const {
    typedef itk::ImageDuplicator< DatasetType > DuplicatorType;
    typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
    duplicator->SetInputImage(d);
    duplicator->Update();
    DatasetPointerType clone = duplicator->GetOutput();
    clone->DisconnectPipeline();
    return clone;
}

} // namespace itk

#endif
