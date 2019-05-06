
#include <intrin.h>
#include <Math/SSE_Swizzle.h>
#include <Common/Gradient.h>

#include <Core/Context.h>
#include <Core/Octree.h>
#include <Core/FileStream.h>
#include <Core/DebugRender.h>
#include <Core/Timer.h>
#include <Core/TreeTraverse.h>

#include <Render/TextureBase.h>
#include <Core/Camera.h>
#include <Common/Image.h>
#include <Render/gl/OpenGLRender.h>





namespace
{
	
#define DEBUG_BORDERS 1
	
	

	inline float minVec4Oct(const Math::Vector4f& input)
	{
		__m128 xyzz = _mm_swizzle_ps_xyzz(input.m_asSSE);
		xyzz = _mm_min_ps(xyzz, _mm_shuffle_ps(xyzz, xyzz, _MM_SHUFFLE(2, 1, 0, 3)));
		xyzz = _mm_min_ps(xyzz, _mm_shuffle_ps(xyzz, xyzz, _MM_SHUFFLE(1, 0, 3, 2)));
		return xyzz.m128_f32[0];
		//float fMin = _mm_min
	}

	inline float maxVec4Oct(const Math::Vector4f& input)
	{
		__m128 xyzz = _mm_swizzle_ps_xyzz(input.m_asSSE);
		xyzz = _mm_max_ps(xyzz, _mm_shuffle_ps(xyzz, xyzz, _MM_SHUFFLE(2, 1, 0, 3)));
		xyzz = _mm_max_ps(xyzz, _mm_shuffle_ps(xyzz, xyzz, _MM_SHUFFLE(1, 0, 3, 2)));
		return xyzz.m128_f32[0];
	}

	inline Math::Vector4f minVec4(const Math::Vector4f& a, const Math::Vector4f& b)
	{
		Math::Vector4f res;
		res.m_asSSE = _mm_min_ps(a.m_asSSE, b.m_asSSE);
		return res;
	}

	inline Math::Vector4f maxVec4(const Math::Vector4f& a, const Math::Vector4f& b)
	{
		Math::Vector4f res;
		res.m_asSSE = _mm_max_ps(a.m_asSSE, b.m_asSSE);
		return res;
	}

	/*
		Assumes tValues are { tMin, tMid, tMax }
	*/
	inline int getTValues(int nodeIdx, const Math::Vector4f* tValues,
		Math::Vector4f& t0Out, Math::Vector4f& t1Out)
	{
		static const int nodeIndices[3 * 8] =
		{
			//x-exit		y-exit		z-exit
			2,			4,			1,		//node index 0
			3,			5,			8,		//node index 1
			8,			6,			3,		//node index 2
			8,			7,			8,		//node index 3
			6,			8,			5,		//node index 4
			7,			8,			8,		//node index 5
			8,			8,			7,		//node index 6
			8,			8,			8		//node index 7
		};

		int startX = 0,			//initialize to t0 values
			startY = 0,
			startZ = 0;

		if (nodeIdx & 1)		//entry is mid[2] exit is tmax[2] //z
			startZ++;
		if ((nodeIdx >> 2) & 1) //entry is mid[1] exit is tmax[1] //y
			startY++;
		if ((nodeIdx >> 1) & 1) //entry is mid[0] exit is tmax[0] //x
			startX++;

		t0Out[Core::OCTREE_X_VAL] = tValues[startX][Core::OCTREE_X_VAL];
		t0Out[Core::OCTREE_Y_VAL] = tValues[startY][Core::OCTREE_Y_VAL];
		t0Out[Core::OCTREE_Z_VAL] = tValues[startZ][Core::OCTREE_Z_VAL];

		t1Out[Core::OCTREE_X_VAL] = tValues[startX + 1][Core::OCTREE_X_VAL];
		t1Out[Core::OCTREE_Y_VAL] = tValues[startY + 1][Core::OCTREE_Y_VAL];
		t1Out[Core::OCTREE_Z_VAL] = tValues[startZ + 1][Core::OCTREE_Z_VAL];


		if (t1Out[Core::OCTREE_X_VAL] < t1Out[Core::OCTREE_Y_VAL] &&
			t1Out[Core::OCTREE_X_VAL] < t1Out[Core::OCTREE_Z_VAL]) return nodeIndices[nodeIdx * 3 + 0]; //YZ plane
		if (t1Out[Core::OCTREE_Y_VAL] < t1Out[Core::OCTREE_X_VAL] &&
			t1Out[Core::OCTREE_Y_VAL] < t1Out[Core::OCTREE_Z_VAL]) return nodeIndices[nodeIdx * 3 + 1]; //XZ plane
		return nodeIndices[nodeIdx * 3 + 2]; // XY plane;

	}

	inline int	firstNode(const float* tMid, const float* t0)
	{
		int result = 0;
		if (t0[Core::OCTREE_X_VAL] > t0[Core::OCTREE_Y_VAL] &&
			t0[Core::OCTREE_X_VAL] > t0[Core::OCTREE_Z_VAL]) //YZ plane
		{
			if (tMid[Core::OCTREE_Y_VAL] < t0[Core::OCTREE_X_VAL]) result |= 4;
			if (tMid[Core::OCTREE_Z_VAL] < t0[Core::OCTREE_X_VAL]) result |= 1;
			return result;
		}
		if (t0[Core::OCTREE_Y_VAL] > t0[Core::OCTREE_X_VAL] &&
			t0[Core::OCTREE_Y_VAL] > t0[Core::OCTREE_Z_VAL]) //XZ plane
		{
			if (tMid[Core::OCTREE_X_VAL] < t0[Core::OCTREE_Y_VAL]) result |= 2;
			if (tMid[Core::OCTREE_Z_VAL] < t0[Core::OCTREE_Y_VAL]) result |= 1;
			return result;
		}
		//XY plane
		if (tMid[Core::OCTREE_X_VAL] < t0[Core::OCTREE_Z_VAL]) result |= 2;
		if (tMid[Core::OCTREE_Y_VAL] < t0[Core::OCTREE_Z_VAL]) result |= 4;

		return result;
	}


	struct OctStack
	{
		OctStack()
		{}

		OctStack(const Math::Vector4f& t0, const Math::Vector4f& t1, int nodeIndex)
			: m_tValues{ t0, (t0 + t1) * 0.5f, t1 }
			, m_nodeIndex(nodeIndex) //index into array
			, m_curIndex(-1)		 //current index[0....7]
		{

		}



		Math::Vector4f m_tValues[3];
		int			   m_nodeIndex;
		int			   m_curIndex;
		float		   m_tMin;

	};


	//Common::Array<Core::SVOTestNode> generateMengerSponge()
	//{
	//	using namespace Core;
	//	//for( int i = 0;)
	//	//Common::Array<Core::SVOTestNode> nodeList;
	//	//SVOTestNode root;
	//	//root.m_childBits = 255; //all bits set for first root
	//	//root.m_firstNode = 1;
	//	//root.m_rgba = Math::Vector4ub( 255, 0, 0, 0 );
	//	
	//}
}

namespace Core
{
	



	Octree::Octree(Context* context) : Component( context )
	{
		
	}

	void Octree::registerObject(Context* context)
	{
		context->RegisterFactory<Octree>();
	}

	Math::BBox3f ChildBounds(const Math::BBox3f& parent, int childIdx)
	{
		static const int indices[3] = { OCTREE_RIGHT, OCTREE_FRONT, OCTREE_TOP };
		
		Math::Vector3f min;
		Math::Vector3f max;
		for (int i = 0; i < 3; ++i)
		{
			if (indices[i] & childIdx)
			{
				min[i] = (parent.m_max[i] + parent.m_min[i]) * 0.5f;
				max[i] = parent.m_max[i];
			}
			else
			{
				min[i] = parent.m_min[i];
				max[i] = (parent.m_max[i] + parent.m_min[i]) * 0.5f;
			}
		}
		return Math::BBox3f(min, max);
	}

	bool HasChild(uint_8 bitSet, int childIdx)
	{
		int idx = 1 << childIdx;
		return (idx & bitSet) != 0;
	}

	void Octree::setBounds(const Math::BBox3f& bounds)
	{
		m_rootNode.m_bounds = bounds;
	}

	const Math::BBox3f& Octree::getBounds() const
	{
		return m_rootNode.m_bounds;
	}
	
	ExtendedOctant* Octree::getRoot() const
	{
		return &m_rootNode;
	}

	void Octree::addBounds_R(const Math::BBox3f& bounds, ExtendedOctant* parentNode)
	{
		if (parentNode->isLeaf()) {
			return;
		}
		
		for (int i = 0; i < 8; ++i)
		{
			auto childBounds = ChildBounds(parentNode->m_bounds, i);
			if (childBounds.instersects(bounds) )
			{
				if (!parentNode->m_childs[i]) {
					parentNode->m_childs[i] = new ExtendedOctant;
					parentNode->m_childs[i]->m_bounds = childBounds;
					parentNode->m_childs[i]->m_depth = parentNode->m_depth + 1;
				}
				addBounds_R(bounds, parentNode->m_childs[i]);
			}
		}
	}



	
	void Octree::addPoint_R(const Math::Vector3f& xyz, ExtendedOctant* parentNode)
	{
		if (parentNode->isLeaf()) {
			return;
		}
		for (int i = 0; i < 8; ++i)
		{
			auto childBounds = ChildBounds(parentNode->m_bounds, i);
			if (childBounds.insideBounds(xyz))
			{
				if (!parentNode->m_childs[i]) {
					parentNode->m_childs[i] = new ExtendedOctant;
					parentNode->m_childs[i]->m_bounds = childBounds;
					parentNode->m_childs[i]->m_depth = parentNode->m_depth + 1;
				}
				addPoint_R(xyz, parentNode->m_childs[i]);
			}
		}
	}

	Common::Array<ExtendedOctant*> Octree::getNodeList(const Math::BBox3f& bounds) const
	{
		Common::Array<ExtendedOctant*> leafList;
		Common::Array<ExtendedOctant*> octantStack;
		if( m_rootNode.m_bounds.instersects( bounds ) )
			octantStack.add(&m_rootNode);
		while (octantStack.getSize())
		{
			auto* octant = octantStack.popLastElement();
			if (octant->isLeaf())
				leafList.add(octant);
			for (int i = 0; i < 8; ++i)
			{
				if ( octant->m_childs[i] && octant->m_childs[i]->m_bounds.instersects( bounds ))
					octantStack.add(octant->m_childs[i]);				
			}
		}
		return leafList;
	}

	
	
		

	SparseOctree::SparseOctree(Context* context) 
		: Component( context )
		//, m_gpuInitialized( false )
		, m_lookupTex( NULL )
		, m_normalTex( NULL )
		, m_rgbaTex( NULL )
		, m_curXLoc(0)
		, m_curYLoc(0)
	{
		//generate empty lookup data
		Common::Array<int_16> lookupData(TREE_IMAGE_DIMS*TREE_IMAGE_DIMS * 3);
		memset(&lookupData[0], 0, sizeof(int_16) * lookupData.getSize());
		//create images here already
		m_lookupImg = new Common::Image("LookupTex");
		m_lookupImg->loadFromMemory(&lookupData[0], Common::IMAGE_FORMAT_RGB16_I_INTEGER, TREE_IMAGE_DIMS, TREE_IMAGE_DIMS);
		//rgba image
		Common::Array<uint_8> rgbaData( TREE_IMAGE_DIMS *TREE_IMAGE_DIMS * 4);
		memset(&rgbaData[0], 0, sizeof( uint_8) * rgbaData.getSize() );
		m_rgbaImg = new Common::Image("RGBA");
		m_rgbaImg->loadFromMemory(&rgbaData[0], Common::IMAGE_FORMAT_RGBA8_UI, TREE_IMAGE_DIMS, TREE_IMAGE_DIMS);
		//onv normal image
		Common::Array<int_16> normalData(TREE_IMAGE_DIMS*TREE_IMAGE_DIMS);
		memset(&normalData[0], 0, sizeof(int_16) * normalData.getSize());
		m_normalImg = new Common::Image("Normal");
		m_normalImg->loadFromMemory(&normalData[0], Common::IMAGE_FORMAT_R16_UI_INTEGER, TREE_IMAGE_DIMS, TREE_IMAGE_DIMS);

		//resize tree node 
		m_treeNodeList.resize(TREE_IMAGE_DIMS * TREE_IMAGE_DIMS);

		auto* renderer = m_context->GetSubsystem<Render::OpenGLRenderer>();
		m_lookupTex = renderer->createTexture();
		m_normalTex = renderer->createTexture();
		m_rgbaTex = renderer->createTexture();

	

	}

	SparseOctree::~SparseOctree()
	{
		delete m_lookupImg;
		delete m_lookupTex;

		delete m_normalImg;
		delete m_normalTex;

		delete m_rgbaImg;
		delete m_rgbaTex;
	}

	void SparseOctree::registerObject(Context* context)
	{
		context->RegisterFactory<SparseOctree>();
	}
	

	bool SparseOctree::loadHeader(const Common::SimpleString& fileName)
	{
		std::ifstream input(fileName.c_string(), std::ios::binary);
		assert(input.is_open());
		input.read((char*)&m_header, sizeof(SVOHeader));

		if (!(m_header.m_magicWord[0] == 'D' && m_header.m_magicWord[1] == 'S' &&
			  m_header.m_magicWord[2] == 'V' && m_header.m_magicWord[3] == 'O'))
			return false;

		if (!(m_header.m_majorVers == MAJOR_VERSION &&
			m_header.m_minorVers == MINOR_VERSION))
			return false;
		m_rootSize = Math::maxVec3(m_header.m_worldBounds.getSize().toConstPointer());
		for (int i = 0; i < 3; ++i) {
			m_header.m_worldBounds.m_min[i] = 0.0f;
			m_header.m_worldBounds.m_max[i] = m_header.m_worldBounds.m_min[i] + m_rootSize;
		}
		m_fileName = fileName; //set filename

		uint_64 nodeOffsets = 0;
		for (int i = 0; i < MAX_LEVELS; ++i)
		{
			uint_64 numPoints = m_header.m_pointsPerLevel[i];
			if (!numPoints)
				break;

			nodeOffsets += m_header.m_pointsPerLevel[i];
			m_cumOffsets.add(nodeOffsets);
		}
		return true;
	}

	


	bool SparseOctree::loadOctree( int maxLevels )
	{
		std::ifstream input(m_fileName.c_string(), std::ios::binary);
		assert(input.is_open());
		input.seekg(sizeof(SVOHeader)); //seek to after file header
		uint_32 index = 0;
		for (int i = 0; i < maxLevels; ++i)
		{
			uint_64 numPoints = m_header.m_pointsPerLevel[i];
			if (!numPoints)
				break;
			Common::Array<InternalNode> nodeList((int_32)numPoints);

			input.read((char*)&nodeList[0], sizeof(InternalNode) * numPoints);
			uint_64 curOffset = m_cumOffsets[i];
			bool isLast = i == maxLevels - 1;
			for (int j = 0; j < numPoints; ++j)
			{
				auto& node = nodeList[j];
				int numChilds = Math::popCount8(node.m_childs);
				SVOTestNode theNode;
				theNode.m_rgba = node.m_rgba;
				
				theNode.m_firstNode = isLast ? -1 : (int)curOffset;
				theNode.m_childBits = isLast ?  0 : node.m_childs;
				//theNode.m_firstNode = isLast ? 0 : curOffset;
				//theNode.m_childBits = isLast ? m_nodeList[0].m_childBits : node.m_childs;
				m_nodeList.add(theNode);
				curOffset += numChilds;
			}
		}

		Common::Array<int_32> borderNodes;
		for (int i = 0; i < (int)m_nodeList.getSize(); ++i)
		{
			const auto& node = m_nodeList[i];
			int numChilds = Math::popCount8(node.m_childBits);
			int oldIdx = node.m_firstNode / TREE_IMAGE_DIMS;
			int newIdx = ( node.m_firstNode + numChilds ) / TREE_IMAGE_DIMS;
			if (newIdx != oldIdx) //children are on the edge of the image, add to list
			{	
				borderNodes.add( i );	
#if DEBUG_BORDERS //draw 'moved' nodes as red in rgba texture
				int firstChild = node.m_firstNode;
				for (int j = 0; j < numChilds; ++j)
				{
					int childIdx = firstChild + j;
					m_nodeList[childIdx].m_rgba = Math::Vector4ub( 255, 0, 0, 255 );
				}
#endif							
			}
		}


		for (int i = 0; i < (int)borderNodes.getSize(); ++i)
		{
			int idx = borderNodes[i];
			auto node = m_nodeList[idx];
			int firstChild = node.m_firstNode;
			int numChilds = Math::popCount8( node.m_childBits );
			//new node offset at the end of the array
			int nodeOffset = m_nodeList.getSize();
			//we're adding new nodes see if we don't cross image border
			//add empty nodes if we do
 			int oldIdx = nodeOffset/TREE_IMAGE_DIMS;
			int newIdx = (nodeOffset+numChilds)/TREE_IMAGE_DIMS;
			if (oldIdx != newIdx)
			{
				int emptyNodes = newIdx*TREE_IMAGE_DIMS - nodeOffset;
				for( int j = 0; j < emptyNodes; ++j )
					m_nodeList.add( SVOTestNode() );
			}

			uint_32 newOffset = m_nodeList.getSize();
			//copy children to the end of the list
			for (int j = 0; j < numChilds; ++j)
				m_nodeList.add( m_nodeList[firstChild +j] );
			//assign new first child index
			m_nodeList[idx].m_firstNode = newOffset;
		}


		//copy node data into image's
		for (int i = 0; i < (int)m_nodeList.getSize(); ++i)
		{
			//rgba image
			Math::Vector4ub* rgbaPtr = reinterpret_cast<Math::Vector4ub*>(m_rgbaImg->getImageData(0));
			rgbaPtr[i] = m_nodeList[i].m_rgba;
			//normal image
			uint_16* normalPtr = reinterpret_cast<uint_16*>(m_normalImg->getImageData(0));
			normalPtr[i] = m_nodeList[i].m_normal;
			//indirection image
			Math::Vector3i16* indirectPtr = reinterpret_cast<Math::Vector3i16*>(m_lookupImg->getImageData(0));
			Math::Vector3i16& curData = indirectPtr[i];
			int xLoc = -1, 
				yLoc = -1;
			int numChilds = Math::popCount8(m_nodeList[i].m_childBits);
			if (numChilds)
			{
				xLoc = m_nodeList[i].m_firstNode % TREE_IMAGE_DIMS;
				yLoc = m_nodeList[i].m_firstNode / TREE_IMAGE_DIMS;
			}			
			
			curData[0] = xLoc;
			curData[1] = yLoc;
			curData[2] = numChilds;			

		}

		m_lookupTex->load(*m_lookupImg, "Nearest");
		m_rgbaTex->load(*m_rgbaImg, "Nearest");
		m_normalTex->load(*m_normalImg, "Nearest");

		

		//std::cout << "Num Nodes Loaded " << m_nodeList.getSize() << "\n";

		return true;
	}

	bool SparseOctree::getAvailableLocation(int numNodes, int& xLoc, int& yLoc)
	{
		static const int NUM_ELEMS = TREE_IMAGE_DIMS * TREE_IMAGE_DIMS;
		if (m_freePosition + numNodes < NUM_ELEMS)
		{

			xLoc = m_freePosition % TREE_IMAGE_DIMS;
			yLoc = m_freePosition / TREE_IMAGE_DIMS;

			m_freePosition += numNodes;

			return true;
		}
		return false;
	}

	bool SparseOctree::getAvailableLocation2(int numNodes, int& xLoc, int& yLoc)
	{
		if (m_curXLoc + numNodes < TREE_IMAGE_DIMS) //space available on this row
		{
			xLoc = m_curXLoc;
			yLoc = m_curYLoc;
			m_curXLoc += numNodes;
			return true;
		}
		else if( m_curYLoc < TREE_IMAGE_DIMS )
		{
			xLoc = 0;
			yLoc = m_curYLoc + 1;
			m_curXLoc = numNodes;
			m_curYLoc = m_curYLoc + 1;
			return true;
		}

		return false;
	}

	int_64 SparseOctree::getNode(OctreeRay& octRay)
	{
		
		static const float ratios[21] =
		{
			1.0 / float(1 << 0),  1.0 / float(1 << 1),  1.0 / float(1 << 2),  1.0 / float(1 << 3),
			1.0 / float(1 << 4),  1.0 / float(1 << 5),  1.0 / float(1 << 6),  1.0 / float(1 << 7),
			1.0 / float(1 << 8),  1.0 / float(1 << 9),  1.0 / float(1 << 10), 1.0 / float(1 << 11),
			1.0 / float(1 << 12), 1.0 / float(1 << 13), 1.0 / float(1 << 14), 1.0 / float(1 << 15),
			1.0 / float(1 << 16), 1.0 / float(1 << 17), 1.0 / float(1 << 18), 1.0 / float(1 << 19),
			1.0 / float(1 << 20)
		};


		using namespace Math;
		const CameraRay<float>& ray = octRay.m_ray;
		Vector4f tMin, tMax, ta, tb;
		float mint, maxt;

		ta = (m_header.m_worldBounds.m_min - ray.m_orig) * ray.m_invRayDir;
		tb = (m_header.m_worldBounds.m_max - ray.m_orig) * ray.m_invRayDir;

		tMin = minVec4(ta, tb); //find min values
		tMax = maxVec4(ta, tb);	//find max values

		mint = maxVec3(tMin.toConstPointer()); //find min entry
		maxt = minVec3(tMax.toConstPointer());

		OctStack travStack[MAX_LEVELS];
		if ((maxt > 0.0f) && (maxt > mint)) //hit
		{
			int curStackIdx = 0;
			travStack[curStackIdx++] = OctStack(tMin, tMax, 0);

			while (curStackIdx) //do a depth first search
			{
				auto& parent = travStack[curStackIdx - 1];
				if (parent.m_curIndex == 8) //miss -> pop from stack and continue
				{
					curStackIdx--;
					continue;
				}

				if (parent.m_curIndex == -1)
				{
					maxt = minVec3(parent.m_tValues[2].toConstPointer());
					octRay.m_dist = maxt;
					if (maxt < 0.0f) //this voxel is behind camera
					{
						curStackIdx--;
						continue;
					}
					float nodeSize = m_rootSize * ratios[curStackIdx - 1];
					float pixSize = nodeSize / maxt;
					if (pixSize < octRay.m_pixScale) //too small
					{
						//return Core::SMALL_NODE;
						//rgba = m_nodeList[parent.m_nodeIndex].m_rgba;
						//rgba[2] = 0xFF;
						return parent.m_nodeIndex;
					}

					parent.m_curIndex = firstNode(parent.m_tValues[1].toConstPointer(),
					parent.m_tValues[0].toConstPointer());
				}

				octRay.m_numIter++; //increment number of iterations for this pixel


				auto& node = m_nodeList[parent.m_nodeIndex];
				if (node.m_childBits == 0)
				{
					//rgba = m_nodeList[parent.m_nodeIndex].m_rgba;
					return parent.m_nodeIndex;
				}

				if (octRay.m_numIter > 1024)
					return INVALID_NODE;

				//if (parent.m_curIndex == -1) //not yet set find entry node
				//if( curStackIdx >= 21 )
				//	return INVALID_NODE;

				Math::Vector4f t0, t1;
				//new child index accounted for negative direction
				int childIdx = parent.m_curIndex ^ octRay.m_xor;
				//update next index
				parent.m_curIndex = getTValues(parent.m_curIndex, parent.m_tValues, t0, t1);

				if ((node.m_childBits >> childIdx) & 1) //push child onto stack
				{
					int childOffset = BitCountBefore(childIdx, node.m_childBits) - 1;
					travStack[curStackIdx++] = OctStack(t0, t1, node.m_firstNode + childOffset);
				}
			}
		}
		return INVALID_NODE; //ray missed the octree
	}

	const SVOHeader& SparseOctree::getHeader() const
	{
		return m_header;
	}

	void SparseOctree::drawTestPlane(Core::Camera* camera, Render::TextureBase* texPtr)
	{
		auto* renderer = m_context->GetSubsystem<Render::OpenGLRenderer>();
		int  FLOAT = renderer->getRenderConstants().m_typeConstants.TC_FLOAT;
		int  UBYTE = renderer->getRenderConstants().m_typeConstants.TC_UNSIGNED_BYTE;
		int  INT = renderer->getRenderConstants().m_typeConstants.TC_INT;
		int  LINES = renderer->getRenderConstants().m_primitiveConstants.PC_LINES;
		int  POINTS = renderer->getRenderConstants().m_primitiveConstants.PC_POINT_LIST;
		int  MAT4X4 = Render::ShaderConstant::FLOAT_MAT4X4;

		Math::Vector3f points[4] = {
			Math::Vector3f(0, 0, 0),
			Math::Vector3f(100, 0, 0),
			Math::Vector3f(100, 100, 0),
			Math::Vector3f(0 ,100, 0)
		};

		Math::Vector2f uvCoords[4] =
		{
			Math::Vector2f(0.0f, 0.0f),
			Math::Vector2f(0.0,  1.0f),
			Math::Vector2f(1.0f, 1.0f),
			Math::Vector2f(1.0f, 0.0f)
		};


		auto wvpMat = camera->getModelView();


		Render::VertexBuffer   vertBuffer((void*)&points[0], FLOAT, 3, 0, 0, "vertIn");
		Render::VertexBuffer   uvBuffer((void*)&uvCoords[0], FLOAT, 2, 0, 0, "uvCoordIn");
		Render::ShaderConstant wvpConstant("wvp", wvpMat.toConstPointer(), 1, MAT4X4, false);

		Render::RenderItem drawItem(renderer);
		drawItem.setPrimitiveType(renderer->getRenderConstants().m_primitiveConstants.PC_QUAD_LIST);
		drawItem.addVertexBuffer(vertBuffer);
		drawItem.addVertexBuffer(uvBuffer);
		drawItem.addShaderConstant(wvpConstant);
		drawItem.setNumVertices(4);
		drawItem.addTexture(texPtr, "tex");
		drawItem.setShader("TextureShader");
		//drawItem.setShader("TextureShaderi16");
		
		renderer->drawRenderItem(drawItem);
	}

	TreeNode::TreeNode()
		: m_frameNum(0)
		, m_rgb565(0)
		, m_normal( 0 )
		, m_childFlags(0)
		, m_xLoc(0)
		, m_yLoc(0)
		, m_childXLoc(0)
		, m_childYLoc(0)
	{

	}

}