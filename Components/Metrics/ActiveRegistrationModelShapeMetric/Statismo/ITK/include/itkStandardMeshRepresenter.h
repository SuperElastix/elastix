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



#ifndef ITK_STANDARD_MESH_REPRESENTER_H
#define ITK_STANDARD_MESH_REPRESENTER_H

#include <boost/unordered_map.hpp>

#include <itkMesh.h>
#include <itkObject.h>

#include "statismoITKConfig.h" // this needs to be the first include file

#include "CommonTypes.h"
#include "Exceptions.h"
#include "Representer.h"

#include "itkPixelConversionTraits.h"

namespace statismo {

template <>
struct RepresenterTraits<itk::Mesh<double, 2u, itk::DefaultStaticMeshTraits<double, 2u, 2u, double, double> > > {

  typedef itk::Mesh<double, 2u, itk::DefaultStaticMeshTraits<double, 2u, 2u, double, double>> MeshType;

  typedef MeshType::Pointer DatasetPointerType;
  typedef MeshType::ConstPointer DatasetConstPointerType;

  typedef MeshType::PointType PointType;
  typedef MeshType::PointType ValueType;
};

template <>
struct RepresenterTraits<itk::Mesh<double, 3u, itk::DefaultStaticMeshTraits<double, 3u, 3u, double, double> > > {

  typedef itk::Mesh<double, 3u, itk::DefaultStaticMeshTraits<double, 3u, 3u, double, double>> MeshType;

  typedef MeshType::Pointer DatasetPointerType;
  typedef MeshType::ConstPointer DatasetConstPointerType;

  typedef MeshType::PointType PointType;
  typedef MeshType::PointType ValueType;
};

template <>
struct RepresenterTraits<itk::Mesh<double, 4u, itk::DefaultStaticMeshTraits<double, 4u, 4u, double, double> > > {

  typedef itk::Mesh<double, 4u, itk::DefaultStaticMeshTraits<double, 4u, 4u, double, double>> MeshType;

  typedef MeshType::Pointer DatasetPointerType;
  typedef MeshType::ConstPointer DatasetConstPointerType;

  typedef MeshType::PointType PointType;
  typedef MeshType::PointType ValueType;
};

}

namespace itk {

// helper function to compute the hash value of an itk point (needed by unorderd_map)
template <typename PointType>
size_t hash_value(const PointType& pt) {
    size_t hash_val = 0;
    for (unsigned i = 0; i < pt.GetPointDimension(); i++) {
        boost::hash_combine( hash_val, pt[i] );
    }
    return hash_val;
}


/**
 * \ingroup Representers
 * \brief A representer for scalar valued itk Meshs
 * \sa Representer
 */
template <class TPixel, unsigned MeshDimension>
class StandardMeshRepresenter : public statismo::Representer<itk::Mesh<TPixel, MeshDimension, itk::DefaultStaticMeshTraits<double, MeshDimension, MeshDimension, double, double> > >, public Object {
  public:

    /* Standard class typedefs. */
    typedef StandardMeshRepresenter            Self;
    typedef Object	Superclass;
    typedef SmartPointer<Self>                Pointer;
    typedef SmartPointer<const Self>          ConstPointer;


    typedef itk::Mesh<TPixel, MeshDimension, itk::DefaultStaticMeshTraits<double, MeshDimension, MeshDimension, double, double>> MeshType;
    typedef typename statismo::Representer<MeshType> RepresenterBaseType;
    typedef typename RepresenterBaseType::DomainType DomainType;
    typedef typename RepresenterBaseType::PointType PointType;
    typedef typename RepresenterBaseType::ValueType ValueType;
    typedef typename RepresenterBaseType::DatasetPointerType DatasetPointerType;
    typedef typename RepresenterBaseType::DatasetConstPointerType DatasetConstPointerType;
    typedef typename MeshType::PointsContainer PointsContainerType;

    /** New macro for creation of through a Smart Pointer. */
    itkSimpleNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( StandardMeshRepresenter, Object );


    static StandardMeshRepresenter* Create() {
        return new StandardMeshRepresenter();
    }

    void Load(const H5::Group& fg);
    StandardMeshRepresenter* Clone() const;

    /// The type of the data set to be used
    typedef MeshType DatasetType;

    // An unordered map is used to cache pointid for corresonding points
    typedef boost::unordered_map<PointType, unsigned> PointCacheType;

    StandardMeshRepresenter();
    virtual ~StandardMeshRepresenter();

    unsigned GetDimensions() const {
        return MeshDimension;
    }
    std::string GetName() const {
        return "itkStandardMeshRepresenter";
    }
    typename RepresenterBaseType::RepresenterDataType GetType() const {
        return RepresenterBaseType::POLYGON_MESH;
    }
    std::string GetVersion() const {
        return "0.1";
    }

    const DomainType& GetDomain() const {
        return m_domain;
    }

    /** Set the reference that is used to build the model */
    void SetReference(DatasetPointerType ds);

    statismo::VectorType PointToVector(const PointType& pt) const;


    /**
     * Converts a sample to its vectorial representation
     */
    statismo::VectorType SampleToSampleVector(DatasetConstPointerType sample) const;

    /**
     * Converts the given sample Vector to a Sample (an itk::Mesh)
     */
    DatasetPointerType SampleVectorToSample(const statismo::VectorType& sample) const;

    /**
     * Returns the value of the sample at the point with the given id.
     */
    ValueType PointSampleFromSample(DatasetConstPointerType sample, unsigned ptid) const;

    /**
     * Given a vector, represening a points convert it to an itkPoint
     */
    ValueType PointSampleVectorToPointSample(const statismo::VectorType& pointSample) const;

    /**
     * Given an itkPoint, convert it to a sample vector
     */
    statismo::VectorType PointSampleToPointSampleVector(const ValueType& v) const;

    /**
     * Save the state of the representer (this simply saves the reference)
     */
    void Save(const H5::Group& fg) const;

    /// return the number of points of the reference
    virtual unsigned GetNumberOfPoints() const;

    /// return the point id associated with the given point
    /// \warning This works currently only for points that are defined on the reference
    virtual unsigned GetPointIdForPoint(const PointType& point) const;

    /// return the reference used in the representer
    DatasetConstPointerType GetReference() const {
        return m_reference;
    }

    void Delete() const {
        this->UnRegister();
    }


    void DeleteDataset(DatasetPointerType d) const { };
    DatasetPointerType CloneDataset(DatasetConstPointerType mesh) const;

  private:

    typename MeshType::Pointer LoadRef(const H5::Group& fg) const;
    typename MeshType::Pointer LoadRefLegacy(const H5::Group& fg) const;

    // returns the closest point for the given mesh
    unsigned FindClosestPoint(const MeshType* mesh, const PointType pt) const ;

    DatasetConstPointerType m_reference;
    DomainType m_domain;
    mutable PointCacheType m_pointCache;
};


} // namespace itk



#include "itkStandardMeshRepresenter.hxx"

#endif /* ITK_STANDARD_MESH_REPRESENTER */
