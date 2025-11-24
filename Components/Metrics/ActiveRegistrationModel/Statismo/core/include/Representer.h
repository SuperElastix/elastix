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
#ifndef REPRESENTER_H_
#define REPRESENTER_H_

#include <string>
#include <itk_H5Cpp.h>

#include "CommonTypes.h"
#include "Domain.h"

/**
 * \brief Provides the interface between statismo and the dataset type the application uses.
 *
 * A Representer is a type that provides the connection between the statismo library
 * and the application. It distinguishes three different representations of the data, and provides methods for conversion between those representations:
 * - a Dataset, typically as read from a file on the disk
 * - a Sample, which is a geometric (generally a rigid or affine) transform of the dataset
 * - a SampleVector, which is an internal representation (vector) useful from the statistical analysis.
 *
 * In the following the methods and types that have to be implemented to write a new
 * Representer for your application are given.
 *
 * \warning This class is never actually used, but serves only for documentation purposes.
 */
//RB: would it be possible to make all representers inherit from it, so as to strictly enforce the interface?
namespace statismo {

template<class T>
class RepresenterTraits {
};

template<class T>
class Representer {

  public:

    enum RepresenterDataType {
        UNKNOWN = 0,
        POINT_SET = 1,
        POLYGON_MESH = 2,
        VOLUME_MESH = 3,
        IMAGE = 4,
        VECTOR = 5,
        CUSTOM = 99
    };

    static RepresenterDataType TypeFromString(const std::string& s) {
        if (s == "POINT_SET")
            return POINT_SET;
        else if (s == "POLYGON_MESH")
            return POLYGON_MESH;
        else if (s == "VOLUME_MESH")
            return VOLUME_MESH;
        else if (s == "IMAGE")
            return IMAGE;
        else if (s == "VECTOR")
            return VECTOR;
        else if (s == "CUSTOM")
            return CUSTOM;
        else
            return UNKNOWN;
    }

    static std::string TypeToString(const RepresenterDataType& type) {
        switch (type) {
        case POINT_SET: {
            return "POINT_SET";
            break;
        }
        case POLYGON_MESH: {
            return "POLYGON_MESH";
            break;
        }
        case VOLUME_MESH: {
            return "VOLUME_MESH";
            break;
        }
        case IMAGE: {
            return "IMAGE";
            break;
        }
        case VECTOR: {
            return "VECTOR";
            break;
        }
        case CUSTOM: {
            return "CUSTOM";
            break;
        }
        default: {
            return "UNKNOWN";
        }
        }
    }

    /**
     * \name Type definitions
     */
    ///@{
    /// Defines (a pointer to) the type of the dataset that is represented.
    /// This could either be a naked pointer or a smart pointer.
    typedef typename RepresenterTraits<T>::DatasetPointerType DatasetPointerType;

    /// Defines the const pointer type o fthe datset that is represented
    typedef typename RepresenterTraits<T>::DatasetConstPointerType DatasetConstPointerType;

    /// Defines the pointtype of the dataset
    typedef typename RepresenterTraits<T>::PointType PointType;

    /// Defines the type of the value when the dataset is evaluated at a given point
    /// (for a image, this could for example be a scalar value or an RGB value)
    typedef typename RepresenterTraits<T>::ValueType ValueType;

    typedef T DatasetType;

    typedef Domain<PointType> DomainType;

    virtual ~Representer() {
    }

    /// Returns a name that identifies the representer
    virtual std::string GetName() const = 0;

    virtual RepresenterDataType GetType() const = 0;

    /// Returns the dimensionality of the dataset (for a mesh this is 3, for a scalar image
    /// this would be 1)
    virtual unsigned GetDimensions() const = 0;
    ///@}

    virtual std::string GetVersion() const = 0;

    /**
     * \name Object creation and destruction
     */
    ///@{
    /** Creates a new representer object, with the
     * the information defined inthe given hdf5 group
     * \sa Save
     */
    virtual void Load(const H5::Group& fg) = 0;

    /** Clone the representer */
    virtual Representer* Clone() const = 0;

    /** Delete the representer object */
    virtual void Delete() const = 0;

    ///@}

    /**
     * \name Adapter methods
     */
    virtual void DeleteDataset(DatasetPointerType d) const = 0;
    virtual DatasetPointerType CloneDataset(DatasetConstPointerType d) const = 0;
    ///@}





    /**
     * \name Conversion from the dataset to a vector representation and back
     */
    ///@{

    /**
     * Returns the Domain for this representers. The domain is essentially a list of all the points on which the model is defined.
     * \sa statismo::Domain
     */
    virtual const statismo::Domain<PointType>& GetDomain() const = 0;


    virtual DatasetConstPointerType GetReference() const = 0;

    /**
     * Converts a Dataset::PointType to a vector in statismo::Vector
     */
    virtual VectorType PointToVector(const PointType& pt) const = 0;

    /**
     * Returns a vectorial representation of the given sample.
     */
    virtual VectorType SampleToSampleVector(
        DatasetConstPointerType sample) const = 0;

    /**
     * Takes a vector of nd elements and converts it to a sample. The sample is a type
     * that is represnter (e.g. an image, a mesh, etc).
     */
    virtual DatasetPointerType SampleVectorToSample(
        const VectorType& sample) const = 0;

    /**
     * Returns the value of the sample at the point with the given id.
     */
    virtual ValueType PointSampleFromSample(DatasetConstPointerType sample,
                                            unsigned ptid) const = 0;

    /**
     * Take a point sample (i.e. the value of a sample at a given point) and converts it
     * to its vector representation.
     * The type of the point sample is a ValueType, that depends on the type of the dataset.
     * For a mesh this would for example be a 3D point,
     * while for a scalar image this would be a scalar value representing the intensity.
     */
    virtual ValueType PointSampleVectorToPointSample(
        const VectorType& v) const = 0;

    /**
     * Convert the given vector represenation of a pointSample back to its ValueType
     * \sa PointSampleVectorToPointSample
     */
    virtual VectorType PointSampleToPointSampleVector(
        const ValueType& pointSample) const = 0;

    /**
     * Defines the mapping between the point ids and the position in the vector.
     * Assume for example that a 3D mesh type is representerd.
     * A conversion strategy used in DatasetToSampleVector could be to return
     * a vector \f$(pt1_x, pt1_y, pt1_z, ..., ptn_x, ptn_y, ptn_z\f$.
     * In this case, this method would return for inputs ptId, componentId
     * the value ptId * 3 + componentId
     */
    virtual unsigned MapPointIdToInternalIdx(unsigned ptId,
            unsigned componentInd) const {
        return ptId * GetDimensions() + componentInd;
    }

    /**
     * Given a point (the coordinates) return the pointId of this point.
     */
    virtual unsigned GetPointIdForPoint(const PointType& point) const = 0;

    ///@}

    /**
     * \name Persistence
     */
    ///@{
    /**
     * Save the informatino that define this representer to the group
     * in the HDF5 file given by fg.
     */
    virtual void Save(const H5::Group& fg) const = 0;

    ///@}

    /**
     * \name Utiities
     */
    /*
     * Returns a new dataset that corresponds to the zero element of the underlying vectorspace
     * obtained when vectorizing a dataset.
     *
     */
    virtual DatasetPointerType IdentitySample() const {

        switch (this->GetType()) {
        case POINT_SET:
        case POLYGON_MESH:
        case VOLUME_MESH: {
            return CloneDataset(this->GetReference());
            break;
        }
        case IMAGE:
        case VECTOR: {
            VectorType zeroVec = VectorType::Zero(GetDomain().GetNumberOfPoints() * GetDimensions());
            return SampleVectorToSample(zeroVec);
            break;
        }
        default: {
            throw statismo::StatisticalModelException(
                "No cannonical identityDataset method is defined for custom Representers.");
        }
        }
    }
};

}

#endif /* REPRESENTER_H_ */

