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

#ifndef ITK_STANDARDIMAGE_REPRESENTER_H_
#define ITK_STANDARDIMAGE_REPRESENTER_H_

#include "statismoITKConfig.h" // this needs to be the first include

#include <itk_H5Cpp.h>

#include <itkObject.h>
#include <itkImage.h>

#include "itkPixelConversionTraits.h"
#include "itkStandardImageRepresenterTraits.h"

#include "CommonTypes.h"
#include "Representer.h"

namespace itk {

/**
 * \ingroup Representers
 * \brief A representer for scalar and vector valued images
 * \sa Representer
 */

template<class TPixel, unsigned ImageDimension>
class StandardImageRepresenter: public Object, public statismo::Representer<
    itk::Image<TPixel, ImageDimension> > {
  public:

    /* Standard class typedefs. */
    typedef StandardImageRepresenter Self;
    typedef Object Superclass;
    typedef SmartPointer<Self> Pointer;
    typedef SmartPointer<const Self> ConstPointer;

    /** New macro for creation of through a Smart Pointer. */
    itkSimpleNewMacro (Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro( StandardImageRepresenter, Object );

    typedef itk::Image<TPixel, ImageDimension> ImageType;

    typedef typename statismo::Representer<ImageType> RepresenterBaseType;
    typedef typename RepresenterBaseType::DomainType DomainType;
    typedef typename RepresenterBaseType::PointType PointType;
    typedef typename RepresenterBaseType::ValueType ValueType;
    typedef typename RepresenterBaseType::DatasetType DatasetType;
    typedef typename RepresenterBaseType::DatasetPointerType DatasetPointerType;
    typedef typename RepresenterBaseType::DatasetConstPointerType DatasetConstPointerType;

    static StandardImageRepresenter* Create() {
        return new StandardImageRepresenter();
    }
    void Load(const H5::Group& fg);
    StandardImageRepresenter* Clone() const;

    StandardImageRepresenter();
    virtual ~StandardImageRepresenter();

    unsigned GetDimensions() const {
        return PixelConversionTrait<TPixel>::GetPixelDimension();
    }
    std::string GetName() const {
        return "itkStandardImageRepresenter";
    }
    typename RepresenterBaseType::RepresenterDataType GetType() const {
        return RepresenterBaseType::IMAGE;
    }

    const DomainType& GetDomain() const {
        return m_domain;
    }
    std::string GetVersion() const {
        return "0.1";
    }

    /// return the reference used in the representer
    DatasetConstPointerType GetReference() const {
        return m_reference;
    }


    /** Set the reference that is used to build the model */
    void SetReference(ImageType* ds);

    /**
     * Creates a sample by first aligning the dataset ds to the reference using Procrustes
     * Alignment.
     */
    statismo::VectorType PointToVector(const PointType& pt) const;
    statismo::VectorType SampleToSampleVector(DatasetConstPointerType sample) const;
    DatasetPointerType SampleVectorToSample(
        const statismo::VectorType& sample) const;

    ValueType PointSampleFromSample(DatasetConstPointerType sample,
                                    unsigned ptid) const;
    ValueType PointSampleVectorToPointSample(
        const statismo::VectorType& pointSample) const;
    statismo::VectorType PointSampleToPointSampleVector(
        const ValueType& v) const;

    void Save(const H5::Group& fg) const;
    virtual unsigned GetPointIdForPoint(const PointType& point) const;

    unsigned GetNumberOfPoints() const;

    void Delete() const {
        this->UnRegister();
    }


    void DeleteDataset(DatasetConstPointerType d) const {}
    DatasetPointerType CloneDataset(DatasetConstPointerType d) const;

  private:

    typename ImageType::Pointer LoadRef(const H5::Group& fg) const;
    typename ImageType::Pointer LoadRefLegacy(const H5::Group& fg) const;

    DatasetConstPointerType m_reference;
    DomainType m_domain;
};

} // namespace itk

#include "itkStandardImageRepresenter.hxx"

#endif /* itkStandardImageREPRESENTER_H_ */
