#ifndef __itkGridScheduleComputer_H__
#define __itkGridScheduleComputer_H__

#include "itkObject.h"
#include "itkVector.h"
#include "itkPoint.h"
#include "itkImageRegion.h"

namespace itk
{
/**
 * \class GridScheduleComputer
 * \brief 
 *
 * \ingroup 
 */

  template < unsigned int VImageDimension >
  class ITK_EXPORT GridScheduleComputer :
	public Object // ProcessObject??
	{
	public:
		
		/** Standard class typedefs. */
		typedef GridScheduleComputer          			Self;
		typedef Object                    					Superclass;
		typedef SmartPointer< Self >								Pointer;
		typedef SmartPointer< const Self >					ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( GridScheduleComputer, Object );

    /** Dimension of the domain space. */
    itkStaticConstMacro( Dimension, unsigned int, VImageDimension );

    /** Typedef's. */
    typedef Point< double, Dimension > 	        OriginType;
    typedef Vector< double, Dimension >         SpacingType;
    typedef ImageRegion< Dimension > 	          RegionType;
    typedef typename RegionType::SizeType       SizeType;
    typedef typename RegionType::SizeValueType  SizeValueType;
    typedef std::vector< OriginType >           VectorOriginType;
    typedef std::vector< SpacingType >          VectorSpacingType;
    typedef std::vector< RegionType >           VectorRegionType;

    /** Set the Origin. */
    itkSetMacro( Origin, OriginType );

    /** Get the Origin. */
    itkGetConstMacro( Origin, OriginType );

    /** Set the Spacing. */
    itkSetMacro( Spacing, SpacingType );

    /** Get the Spacing. */
    itkGetConstMacro( Spacing, SpacingType );

    /** Set the Region. */
    itkSetMacro( Region, RegionType );

    /** Get the Region. */
    itkGetConstMacro( Region, RegionType );

    /** Set the B-spline order. */
    itkSetClampMacro( BSplineOrder, unsigned int, 0, 5 );

    /** Get the B-spline order. */
    itkGetConstMacro( BSplineOrder, unsigned int );

    /** Set a default grid spacing schedule. */
    virtual void SetDefaultGridSpacingSchedule(
      const unsigned int & levels,
      const SpacingType & finalGridSpacing,
      const float & upsamplingFactor );

    /** Set a grid spacing schedule. */
    virtual void SetGridSpacingSchedule(
      const VectorSpacingType & schedule );

    /** Get the grid spacing schedule. */
    virtual void GetGridSpacingSchedule( VectorSpacingType & schedule ) const;

    /** Compute the B-spline grid. */
    virtual void ComputeBSplineGrid( void );

    /** Get the B-spline grid at some level. */
    virtual void GetBSplineGrid( unsigned int level,
      RegionType & gridRegion,
      SpacingType & gridSpacing,
      OriginType & gridOrigin );

    /** This function determines if upsampling is required. */
    virtual bool GetDoUpsampling( const unsigned int & level ) const;

	protected:

    /** The constructor. */
		GridScheduleComputer();

    /** The destructor. */
		virtual ~GridScheduleComputer() {};

    /** PrintSelf. */
		void PrintSelf( std::ostream& os, Indent indent ) const;
		
	private:

		GridScheduleComputer( const Self& );	// purposely not implemented
		void operator=( const Self& );				// purposely not implemented
		
		/** Declare member variables, needed in functions. */
    OriginType          m_Origin;
    SpacingType         m_Spacing;
    RegionType          m_Region;
    unsigned int        m_BSplineOrder;
    unsigned int        m_NumberOfLevels;
    VectorSpacingType   m_GridSpacingSchedule;
    std::vector<bool>   m_DoUpsampling;

    /** Clamp the upsampling factor. */
    itkSetClampMacro( UpsamplingFactor, float, 1.0, NumericTraits<float>::max() );

    /** Declare member variables, needed internally. */
    float               m_UpsamplingFactor;
    bool                m_GridSpacingScheduleIsDownwardsDivisible;

    /** Declare member variables, needed for B-spline grid. */
    VectorOriginType    m_GridOrigins;
    VectorRegionType    m_GridRegions;

	}; // end class GridScheduleComputer
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGridScheduleComputer.txx"
#endif

#endif // end #ifndef __itkGridScheduleComputer_H__

